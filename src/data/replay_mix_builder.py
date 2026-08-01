"""Build training data mixtures with English/code replay buffers.

During continued pretraining (CPT) on Portuguese data, the model risks
"catastrophic forgetting" of English and general capabilities. This module
implements data mixing strategies to mitigate this:

Strategy: Mix Portuguese CPT data with small proportions of English and
code data from the model's original training distribution. This acts as
a "replay buffer" that reminds the model of its original capabilities.

Typical mixture ratios:
- pt_only: 100% Aurora-PT (baseline, no replay)
- pt_en: 85% Aurora-PT + 15% English (FineWeb-Edu)
- pt_en_code: 80% Aurora-PT + 15% English + 5% code (StarCoder)

The English replay data comes from FineWeb-Edu (high-quality educational
web text) and code from StarCoderData (permissively licensed code).

Usage:
    from src.data.replay_mix_builder import ReplayMixBuilder

    builder = ReplayMixBuilder(config["data"])
    mixed_dataset = builder.build_mixture("pt_en", primary_dataset)
"""

from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReplayMixBuilder:
    """Build data mixtures with replay buffers for catastrophic forgetting prevention.

    Takes a primary Portuguese dataset and mixes it with English/code replay
    data according to predefined ratios. The resulting dataset is shuffled
    to ensure even distribution during training.

    Args:
        config: Data config dict containing 'mixtures', 'english_replay',
                and 'code_replay' sections.

    Attributes:
        mixtures: Dict mapping mixture names to source:ratio dicts.
                  Example: {"pt_en": {"aurora_pt": 0.85, "english_replay": 0.15}}
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.mixtures = config.get("mixtures", {})
        self.packing_cfg = config.get("packing", {})
        # Semente para shuffle: usa a configurada em dataset.seed (ou
        # experiment.seed como alternativa), nunca um valor fixo hardcoded.
        self.seed = config.get("dataset", {}).get(
            "seed", config.get("experiment", {}).get("seed", 42)
        )
        # Modo de fallback "silencioso" para replay: por padrão é False, ou
        # seja, falhas ao carregar replay (rede, dataset gated, etc.) levantam
        # exceção em vez de degradar silenciosamente para ~0% de replay.
        self.allow_replay_fallback = config.get("allow_replay_fallback", False)

    def build_mixture(
        self,
        mixture_name: str,
        primary_dataset: Dataset,
        max_tokens: int | None = None,
    ) -> Dataset:
        """Build a specific mixture by name.

        Uses ALL primary data and computes replay sizes so the final mixture
        has the desired proportions. For example, if ratios are
        {"aurora_pt": 0.85, "english_replay": 0.15} and primary has 10000 samples,
        the final mixture will be 10000 PT + 1765 EN ≈ 85%/15% split.

        Args:
            mixture_name: Key in self.mixtures (e.g., "pt_en", "pt_en_code").
            primary_dataset: The main Portuguese dataset (Aurora-PT).
            max_tokens: Optional token budget to cap the final mixture size.
                        Uses a 4 chars/token heuristic for estimation.

        Returns:
            Shuffled HuggingFace Dataset combining all sources.

        Raises:
            ValueError: If mixture_name is not in the configured mixtures.
        """
        if mixture_name not in self.mixtures:
            raise ValueError(
                f"Unknown mixture: {mixture_name}. Available: {list(self.mixtures.keys())}"
            )

        ratios = self.mixtures[mixture_name]
        logger.info(f"Building mixture '{mixture_name}': {ratios}")

        datasets_to_mix = []
        total_primary = len(primary_dataset)

        # Calculate replay sizes relative to primary to achieve target ratios.
        # If aurora_pt ratio is 0.85, then primary represents 85% of the final mix.
        # So total_final = total_primary / pt_ratio, and each other source gets
        # its ratio * total_final samples.
        pt_ratio = ratios.get("aurora_pt", 1.0)
        total_final = total_primary / pt_ratio if pt_ratio > 0 else total_primary

        for source, ratio in ratios.items():
            if source == "aurora_pt":
                # Use ALL primary data (don't truncate it). Project down to
                # just the "text" column so its schema matches the replay
                # datasets below (which only have "text") — otherwise
                # concatenate_datasets() raises on feature/schema mismatch.
                ds = primary_dataset.select_columns(["text"])
                datasets_to_mix.append(ds)
                logger.info(f"  {source}: {total_primary} samples (ratio={ratio})")

            elif source == "english_replay":
                # English replay sized to achieve target proportion in final mix
                n_samples = int(total_final * ratio)
                ds = self._load_english_replay(n_samples)
                ds = self._project_to_text_column(ds)
                datasets_to_mix.append(ds)
                logger.info(f"  {source}: {len(ds)} samples (target={n_samples}, ratio={ratio})")

            elif source == "code":
                # Code replay sized to achieve target proportion
                n_samples = int(total_final * ratio)
                ds = self._load_code_replay(n_samples)
                ds = self._project_to_text_column(ds)
                datasets_to_mix.append(ds)
                logger.info(f"  {source}: {len(ds)} samples (target={n_samples}, ratio={ratio})")

        # Concatenate all sources and shuffle for even distribution
        mixed = concatenate_datasets(datasets_to_mix)
        mixed = mixed.shuffle(seed=self.seed)

        # Optional: cap mixture size by estimated token count
        if max_tokens:
            # Heuristic: ~4 characters per token on average
            est_chars = max_tokens * 4
            cumulative = 0
            cutoff = len(mixed)
            for i in range(len(mixed)):
                cumulative += len(mixed[i]["text"])
                if cumulative >= est_chars:
                    cutoff = i + 1
                    break
            mixed = mixed.select(range(cutoff))

        logger.info(f"Final mixture size: {len(mixed)} samples")
        return mixed

    @staticmethod
    def _project_to_text_column(ds: Dataset) -> Dataset:
        """Drop every column except "text" so schemas match across sources.

        Replay datasets are normalized to a single "text" column, but some
        sources (or future changes) could carry extra columns. Concatenating
        datasets with mismatched features raises ValueError, so we defensively
        project every source down to just "text" right before mixing.
        """
        extra_columns = [c for c in ds.column_names if c != "text"]
        if extra_columns:
            ds = ds.remove_columns(extra_columns)
        return ds

    def _load_english_replay(self, n_samples: int) -> Dataset:
        """Load English replay data from FineWeb-Edu.

        Uses streaming to avoid downloading the full dataset (10B+ tokens).
        Takes only the first n_samples documents.

        Args:
            n_samples: Number of English documents to load.

        Returns:
            Dataset with a "text" column containing English documents.

        Raises:
            RuntimeError: If loading fails (network error, gating, etc.) and
                `allow_replay_fallback` is not enabled in the config. Silently
                degrading a configured replay ratio (e.g. 15% English) down to
                ~0% is a training-data-integrity issue, so failures are loud
                by default. Set `allow_replay_fallback: true` in the data
                config to opt into the old soft-fail behavior (an empty
                placeholder dataset) instead.
        """
        en_cfg = self.config.get("english_replay", {})
        hub_id = en_cfg.get("hub_id", "HuggingFaceFW/fineweb-edu")
        subset = en_cfg.get("subset", "sample-10BT")

        logger.info(f"Loading English replay from {hub_id}/{subset}")
        try:
            # Stream to avoid full download
            ds = load_dataset(hub_id, subset, split="train", streaming=True)
            samples = []
            for i, example in enumerate(ds):
                if i >= n_samples:
                    break
                samples.append({"text": example["text"]})
            return Dataset.from_list(samples)
        except Exception as e:
            message = (
                f"Failed to load English replay data from hub_id="
                f"'{hub_id}' (subset='{subset}'): {e!r}. This mixture "
                "requires English replay data to hit its configured ratio; "
                "proceeding would silently degrade the replay proportion to "
                "~0%."
            )
            if self.allow_replay_fallback:
                logger.warning(
                    f"{message} allow_replay_fallback=true, so falling back "
                    "to an empty placeholder dataset instead of raising."
                )
                return Dataset.from_list([{"text": ""}])
            logger.error(message)
            raise RuntimeError(message) from e

    def _load_code_replay(self, n_samples: int) -> Dataset:
        """Load code replay data from StarCoderData.

        Loops over ALL languages configured in code_replay.languages (not
        just the first), streaming each language subset and concatenating
        the results, so the full configured language mix is represented.

        Args:
            n_samples: Total number of code documents to load, split evenly
                across the configured languages.

        Returns:
            Dataset with a "text" column containing code snippets.

        Raises:
            RuntimeError: If loading fails (network error, gating, etc.) and
                `allow_replay_fallback` is not enabled in the config. See
                `_load_english_replay` for rationale — silent degradation of
                the code replay ratio is worse than a loud failure.
        """
        code_cfg = self.config.get("code_replay", {})
        hub_id = code_cfg.get("hub_id", "bigcode/starcoderdata")
        languages = code_cfg.get("languages", ["python"])
        n_per_language = n_samples // max(len(languages), 1) if n_samples > 0 else 0

        logger.info(f"Loading code replay from {hub_id} (languages={languages})")
        try:
            samples = []
            for language in languages:
                if n_per_language <= 0:
                    continue
                ds = load_dataset(hub_id, data_dir=language, split="train", streaming=True)
                for i, example in enumerate(ds):
                    if i >= n_per_language:
                        break
                    # StarCoderData uses "content" field, fallback to "text"
                    content = example.get("content", example.get("text", ""))
                    samples.append({"text": content})
            return Dataset.from_list(samples[:n_samples] if n_samples else samples)
        except Exception as e:
            message = (
                f"Failed to load code replay data from hub_id='{hub_id}' "
                f"(languages={languages}): {e!r}. Note this dataset is "
                "gated on HuggingFace Hub and requires authentication with "
                "granted dataset access. This mixture requires code replay "
                "data to hit its configured ratio; proceeding would "
                "silently degrade the replay proportion to ~0%."
            )
            if self.allow_replay_fallback:
                logger.warning(
                    f"{message} allow_replay_fallback=true, so falling back "
                    "to an empty placeholder dataset instead of raising."
                )
                return Dataset.from_list([{"text": ""}])
            logger.error(message)
            raise RuntimeError(message) from e
