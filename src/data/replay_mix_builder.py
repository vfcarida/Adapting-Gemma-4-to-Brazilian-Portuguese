"""
src/data/replay_mix_builder.py
──────────────────────────────
Build mixed training data: X% PT-BR (Aurora-PT) + Y% EN high-quality
(e.g. FineWeb-Edu) for replay-based continued pretraining.

Interleaves two streaming sources with proportional sampling and
produces packed sequences compatible with CausalLM training.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

from transformers import PreTrainedTokenizerBase

from src.data.aurora_loader import AuroraLoader, PackedSequenceDataset
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReplayMixBuilder:
    """Create a mixed-language training stream.

    Interleaves documents from a Portuguese source (Aurora-PT) and an
    English source with configurable proportions.

    Parameters
    ----------
    pt_ratio : float
        Proportion of Portuguese documents (e.g. 0.85 for 85%).
    en_ratio : float
        Proportion of English documents (e.g. 0.15 for 15%).
    pt_dataset_id : str
        HuggingFace ID for the Portuguese corpus.
    en_dataset_id : str
        HuggingFace ID for the English corpus.
    pt_text_column : str
        Text column in the Portuguese dataset.
    en_text_column : str
        Text column in the English dataset.
    seed : int
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        pt_ratio: float = 0.85,
        en_ratio: float = 0.15,
        code_ratio: float = 0.0,
        pt_dataset_id: str = "Itau-Unibanco/Aurora-PT",
        en_dataset_id: str = "HuggingFaceFW/fineweb-edu",
        code_dataset_id: str = "bigcode/starcoderdata",
        pt_text_column: str = "text",
        en_text_column: str = "text",
        code_text_column: str = "content",
        seed: int = 42,
        num_shards: int = 1,
        shard_index: int = 0,
    ) -> None:
        if abs(pt_ratio + en_ratio + code_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"pt_ratio ({pt_ratio}) + en_ratio ({en_ratio}) + code_ratio ({code_ratio}) must equal 1.0"
            )

        self.pt_ratio = pt_ratio
        self.en_ratio = en_ratio
        self.code_ratio = code_ratio
        self.seed = seed

        # ⚠️ REQUIRES HF_TOKEN for gated dataset access (Aurora-PT)
        self._pt_loader = AuroraLoader(
            dataset_id=pt_dataset_id,
            text_column=pt_text_column,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        self._en_dataset_id = en_dataset_id
        self._en_text_column = en_text_column
        self._code_dataset_id = code_dataset_id
        self._code_text_column = code_text_column

    def _en_stream(self) -> Iterator[str]:
        """Stream English documents from the replay source."""
        from datasets import load_dataset

        logger.info("Streaming EN replay source: %s", self._en_dataset_id)
        ds = load_dataset(
            self._en_dataset_id,
            split="train",
            streaming=True,
        )
        for example in ds:
            text = example.get(self._en_text_column, "")
            if text and text.strip():
                yield text.strip()

    def _code_stream(self) -> Iterator[str]:
        """Stream Code documents from the replay source."""
        from datasets import load_dataset

        if self.code_ratio <= 0.0:
            return

        logger.info("Streaming CODE replay source: %s", self._code_dataset_id)
        ds = load_dataset(
            self._code_dataset_id,
            split="train",
            streaming=True,
        )
        for example in ds:
            text = example.get(self._code_text_column, "")
            if text and text.strip():
                yield text.strip()

    def stream(self) -> Iterator[str]:
        """Yield mixed PT/EN/CODE documents according to configured ratios.

        Uses probabilistic sampling: for each document slot, draw from
        the Portuguese stream, English stream, or Code stream.
        """
        rng = random.Random(self.seed)
        pt_iter = self._pt_loader.stream()
        en_iter = self._en_stream()
        code_iter = self._code_stream()

        pt_exhausted = False
        en_exhausted = False
        code_exhausted = self.code_ratio <= 0.0

        while True:
            if pt_exhausted and en_exhausted and code_exhausted:
                break

            # Probabilistic source selection
            roll = rng.random()

            if roll < self.pt_ratio and not pt_exhausted:
                try:
                    yield next(pt_iter)
                except StopIteration:
                    pt_exhausted = True
                    logger.info("Portuguese stream exhausted.")
            elif roll < (self.pt_ratio + self.en_ratio) and not en_exhausted:
                try:
                    yield next(en_iter)
                except StopIteration:
                    en_exhausted = True
                    logger.info("English stream exhausted.")
            elif not code_exhausted:
                try:
                    yield next(code_iter)
                except StopIteration:
                    code_exhausted = True
                    logger.info("Code stream exhausted.")
            elif not en_exhausted:
                try:
                    yield next(en_iter)
                except StopIteration:
                    en_exhausted = True
            elif not pt_exhausted:
                try:
                    yield next(pt_iter)
                except StopIteration:
                    pt_exhausted = True

    def build_packed_dataset(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 4096,
    ) -> PackedSequenceDataset:
        """Build a packed-sequence dataset from the mixed stream.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            Model tokenizer.
        max_seq_len : int
            Packed sequence length.

        Returns
        -------
        PackedSequenceDataset
        """
        return PackedSequenceDataset(
            text_iterator=self.stream(),
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )


def build_replay_mix_from_config(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> PackedSequenceDataset:
    """Factory: build a replay mix dataset from YAML config.

    Parameters
    ----------
    config : dict
        Full config dict (expects ``data`` section).
    tokenizer : PreTrainedTokenizerBase
        Model tokenizer.

    Returns
    -------
    PackedSequenceDataset
    """
    data_cfg = config.get("data", config)
    builder = ReplayMixBuilder(
        pt_ratio=data_cfg.get("pt_ratio", 0.85),
        en_ratio=data_cfg.get("en_ratio", 0.15),
        code_ratio=data_cfg.get("code_ratio", 0.0),
        pt_dataset_id=data_cfg.get("dataset_id", "Itau-Unibanco/Aurora-PT"),
        en_dataset_id=data_cfg.get("en_dataset_id", "HuggingFaceFW/fineweb-edu"),
        code_dataset_id=data_cfg.get("code_dataset_id", "bigcode/starcoderdata"),
        num_shards=data_cfg.get("num_shards", 1),
        shard_index=data_cfg.get("shard_index", 0),
    )
    return builder.build_packed_dataset(
        tokenizer=tokenizer,
        max_seq_len=data_cfg.get("max_seq_len", 4096),
    )
