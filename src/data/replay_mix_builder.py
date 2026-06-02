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

    def build_mixture(
        self,
        mixture_name: str,
        primary_dataset: Dataset,
        max_tokens: int | None = None,
    ) -> Dataset:
        """Build a specific mixture by name.

        Computes the number of samples for each source based on the ratio
        relative to the primary dataset size. For example, if primary has
        10000 samples and English ratio is 0.15, loads ~1500 English samples.

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

        for source, ratio in ratios.items():
            if source == "aurora_pt":
                # Primary Portuguese data — take proportion of full dataset
                n_samples = int(total_primary * ratio)
                ds = primary_dataset.select(range(min(n_samples, total_primary)))
                datasets_to_mix.append(ds)
                logger.info(f"  {source}: {len(ds)} samples (ratio={ratio})")

            elif source == "english_replay":
                # English replay from FineWeb-Edu (educational web text)
                n_samples = int(total_primary * ratio)
                ds = self._load_english_replay(n_samples)
                datasets_to_mix.append(ds)
                logger.info(f"  {source}: {len(ds)} samples (ratio={ratio})")

            elif source == "code":
                # Code replay from StarCoderData
                n_samples = int(total_primary * ratio)
                ds = self._load_code_replay(n_samples)
                datasets_to_mix.append(ds)
                logger.info(f"  {source}: {len(ds)} samples (ratio={ratio})")

        # Concatenate all sources and shuffle for even distribution
        mixed = concatenate_datasets(datasets_to_mix)
        mixed = mixed.shuffle(seed=42)

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

    def _load_english_replay(self, n_samples: int) -> Dataset:
        """Load English replay data from FineWeb-Edu.

        Uses streaming to avoid downloading the full dataset (10B+ tokens).
        Takes only the first n_samples documents.

        Args:
            n_samples: Number of English documents to load.

        Returns:
            Dataset with a "text" column containing English documents.
            Returns a single empty-text dataset on failure (graceful degradation).
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
            logger.warning(f"Failed to load English replay: {e}. Using empty dataset.")
            return Dataset.from_list([{"text": ""}])

    def _load_code_replay(self, n_samples: int) -> Dataset:
        """Load code replay data from StarCoderData.

        Loads Python code by default (configurable via code_replay.languages).
        Uses streaming for memory efficiency.

        Args:
            n_samples: Number of code documents to load.

        Returns:
            Dataset with a "text" column containing code snippets.
            Returns a single empty-text dataset on failure (graceful degradation).
        """
        code_cfg = self.config.get("code_replay", {})
        hub_id = code_cfg.get("hub_id", "bigcode/starcoderdata")
        languages = code_cfg.get("languages", ["python"])

        logger.info(f"Loading code replay from {hub_id}")
        try:
            ds = load_dataset(hub_id, data_dir=languages[0], split="train", streaming=True)
            samples = []
            for i, example in enumerate(ds):
                if i >= n_samples:
                    break
                # StarCoderData uses "content" field, fallback to "text"
                content = example.get("content", example.get("text", ""))
                samples.append({"text": content})
            return Dataset.from_list(samples)
        except Exception as e:
            logger.warning(f"Failed to load code replay: {e}. Using empty dataset.")
            return Dataset.from_list([{"text": ""}])
