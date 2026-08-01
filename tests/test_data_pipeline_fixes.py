"""Regression tests that import the REAL src.data.* code (not local
reimplementations) to exercise the bug fixes made to the data pipeline:

- src.data.instruction_data_builder.format_gemma4_chat (no duplicate BOS)
- src.data.replay_mix_builder.ReplayMixBuilder (schema projection + seed)
- src.data.cluster_dedup.build_clusters / split_by_clusters (word shingles,
  test_ratio threaded through)
- src.data.quality_manifest.QualityManifest (index alignment across empty docs)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset

from src.data.cluster_dedup import ClusterDedup
from src.data.instruction_data_builder import format_gemma4_chat
from src.data.quality_manifest import QualityManifest
from src.data.replay_mix_builder import ReplayMixBuilder


class TestFormatGemma4ChatReal:
    """Exercise the real (not locally-reimplemented) format_gemma4_chat."""

    def test_no_hardcoded_bos_prefix(self):
        messages = [{"role": "user", "content": "Oi"}]
        result = format_gemma4_chat(messages)
        # The tokenizer is responsible for BOS via add_special_tokens=True;
        # a literal "<bos>" here would cause a duplicate BOS at tokenization.
        assert not result.startswith("<bos>")
        # Real Gemma 4 turn marker (verified live against the tokenizer's
        # own chat template) — not "<start_of_turn>", which is Gemma 2/3's.
        assert result.startswith("<|turn>user\n")

    def test_still_formats_turns_correctly(self):
        messages = [
            {"role": "user", "content": "Oi"},
            {"role": "model", "content": "Olá!"},
        ]
        result = format_gemma4_chat(messages)
        assert "<|turn>user\nOi<turn|>\n" in result
        assert "<|turn>model\nOlá!<turn|>\n" in result


class TestReplayMixBuilderReal:
    """Exercise the real ReplayMixBuilder (no network calls required for
    the "pt_only" mixture, which only touches the primary dataset)."""

    def test_projects_primary_dataset_to_text_column(self):
        # Primary dataset retains extra HF columns (e.g. from Aurora-PT),
        # which used to raise a schema-mismatch ValueError when concatenated
        # with replay datasets that only have "text".
        primary = Dataset.from_list(
            [
                {"text": "documento um", "domain": "geral", "extra": 1},
                {"text": "documento dois", "domain": "geral", "extra": 2},
            ]
        )
        config = {"mixtures": {"pt_only": {"aurora_pt": 1.0}}, "dataset": {"seed": 123}}
        builder = ReplayMixBuilder(config)

        mixed = builder.build_mixture("pt_only", primary)

        assert mixed.column_names == ["text"]
        assert len(mixed) == 2
        assert set(mixed["text"]) == {"documento um", "documento dois"}

    def test_seed_threaded_from_config_not_hardcoded(self):
        config_a = {"mixtures": {"pt_only": {"aurora_pt": 1.0}}, "dataset": {"seed": 7}}
        config_b = {"mixtures": {"pt_only": {"aurora_pt": 1.0}}, "dataset": {"seed": 999}}
        assert ReplayMixBuilder(config_a).seed == 7
        assert ReplayMixBuilder(config_b).seed == 999
        # Falls back to experiment.seed, then 42, never a bare hardcoded value
        # silently overriding an explicitly configured seed.
        config_c = {"mixtures": {}, "experiment": {"seed": 55}}
        assert ReplayMixBuilder(config_c).seed == 55

    def test_replay_load_failure_raises_loudly_by_default(self):
        # english_replay points at a bogus hub_id, so loading fails. Without
        # allow_replay_fallback=True, this must raise instead of silently
        # degrading the configured replay ratio to ~0%.
        primary = Dataset.from_list([{"text": "doc"} for _ in range(10)])
        config = {
            "mixtures": {"pt_en": {"aurora_pt": 0.5, "english_replay": 0.5}},
            "english_replay": {"hub_id": "this-hub-id/does-not-exist-12345"},
        }
        builder = ReplayMixBuilder(config)
        with pytest.raises(RuntimeError):
            builder.build_mixture("pt_en", primary)


class TestClusterDedupReal:
    """Exercise the real ClusterDedup with word-level shingles."""

    def test_build_clusters_groups_near_duplicates(self):
        dedup = ClusterDedup()
        texts = [
            "o rato roeu a roupa do rei de roma durante toda a noite escura e fria",
            # near-duplicate: only the last word differs (~85% word-shingle Jaccard)
            "o rato roeu a roupa do rei de roma durante toda a noite escura e gelada",
            "completamente outro assunto sem nenhuma relacao aqui",
        ]
        cluster_map = dedup.build_clusters(texts, threshold=0.5, num_perm=64)
        assert cluster_map[0] == cluster_map[1]
        assert cluster_map[2] != cluster_map[0]

    def test_split_by_clusters_respects_test_ratio(self):
        dedup = ClusterDedup()
        # 20 distinct singleton clusters so val/test ratios have enough
        # granularity to produce a non-empty test split.
        texts = [f"documento numero {i} com conteudo unico" for i in range(20)]
        dedup.build_clusters(texts, threshold=0.9, num_perm=64)
        split_map = dedup.split_by_clusters(val_ratio=0.1, test_ratio=0.1, seed=42)
        assert "test" in split_map
        assert len(split_map["test"]) > 0
        # No overlap between splits
        train, val, test = (
            set(split_map["train"]),
            set(split_map["validation"]),
            set(split_map["test"]),
        )
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)


class TestQualityManifestIndexAlignment:
    """Exercise the real QualityManifest to verify filter_by_quality returns
    raw-dataset indices even when empty documents are skipped mid-dataset."""

    def test_filtered_indices_are_raw_dataset_positions(self):
        # Index 1 is an empty document that build_manifest silently skips.
        raw_dataset = [
            {
                "text": "este e um texto valido em portugues com bastante conteudo para passar no filtro de tamanho minimo de caracteres exigido pela configuracao padrao"
            },
            {"text": ""},
            {
                "text": "este e outro texto valido em portugues com bastante conteudo para passar no filtro de tamanho minimo de caracteres exigido pela configuracao padrao"
            },
        ]
        manifest = QualityManifest()
        manifest.build_manifest(raw_dataset)

        # Only 2 records were built (index 1 was skipped), but raw_index
        # must still point back at the correct positions in raw_dataset.
        assert len(manifest.records) == 2
        assert [r.raw_index for r in manifest.records] == [0, 2]

        filtered = manifest.filter_by_quality(min_score=0.0)
        # filtered indices must be raw_dataset positions (0 and 2), NOT
        # positions in the filtered/records list (which would be 0 and 1).
        assert filtered == [0, 2]
        assert raw_dataset[filtered[0]]["text"] != ""
        assert raw_dataset[filtered[1]]["text"] != ""
