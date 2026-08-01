"""Tests for eval task load_data() schema normalization.

Every task file here maps a real (previously-verified) HF dataset schema
onto the flat dict shape the rest of the eval pipeline expects. That
mapping logic — field renames, label encoding (int vs bool vs string),
choices-dict flattening, masked-test-split workarounds — had zero direct
test coverage before this file (only math_pt/xlsum_pt did, added in a
prior pass). Each test mocks `_load_from_hub` (or `datasets.load_dataset`
for the two tasks that call it directly) with a minimal fake row shaped
like the real dataset, and asserts the normalized output — no network.
"""

from src.eval.tasks.assin2_rte import Assin2RTETask
from src.eval.tasks.assin2_sts import Assin2STSTask
from src.eval.tasks.bluex import BluexTask
from src.eval.tasks.boolq_pt import BoolQPT
from src.eval.tasks.broverbs import BRoverbsTask
from src.eval.tasks.copa_pt import CopaPTTask
from src.eval.tasks.enem import EnemTask
from src.eval.tasks.hatebr import HateBRTask
from src.eval.tasks.legalbench_br import LegalBenchBR
from src.eval.tasks.lener_br import LeNERBr
from src.eval.tasks.mrpc_pt import MRPCPTTask
from src.eval.tasks.oab_bench import OABBenchTask
from src.eval.tasks.publichearing_br import PublicHearingBR
from src.eval.tasks.rte_pt import RTEPTTask
from src.eval.tasks.tweet_sentbr import TweetSentBRTask


class TestAssin2RTELoadData:
    def test_entailment_judgment_mapped_to_label(self, monkeypatch):
        task = Assin2RTETask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "premise": "O gato dorme",
                    "hypothesis": "O animal descansa",
                    "entailment_judgment": 1,
                },
                {
                    "premise": "O gato dorme",
                    "hypothesis": "O cachorro late",
                    "entailment_judgment": 0,
                },
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"premise": "O gato dorme", "hypothesis": "O animal descansa", "label": "entailment"},
            {"premise": "O gato dorme", "hypothesis": "O cachorro late", "label": "not_entailment"},
        ]


class TestAssin2STSLoadData:
    def test_relatedness_score_extracted_as_float(self, monkeypatch):
        task = Assin2STSTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {"premise": "A", "hypothesis": "B", "relatedness_score": 4.5},
            ],
        )
        examples = task.load_data({})
        assert examples == [{"sentence1": "A", "sentence2": "B", "score": 4.5}]
        assert isinstance(examples[0]["score"], float)


class TestBluexLoadData:
    def test_choices_dict_flattened_and_ordered(self, monkeypatch):
        task = BluexTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "question": "Qual a capital?",
                    "choices": {"text": ["Rio", "Brasília", "SP"], "label": ["C", "A", "B"]},
                    "answerKey": "A",
                }
            ],
        )
        examples = task.load_data({})
        # Reordered by label A, B, C -> Brasília(A), SP(B), Rio(C)
        assert examples == [
            {"question": "Qual a capital?", "options": ["Brasília", "SP", "Rio"], "answer": "A"}
        ]


class TestBoolQPTLoadData:
    def test_int_label_converted_to_bool_answer(self, monkeypatch):
        task = BoolQPT()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {"passage": "Texto...", "question": "É verdade?", "label": 1},
                {"passage": "Texto...", "question": "É falso?", "label": 0},
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"passage": "Texto...", "question": "É verdade?", "answer": True},
            {"passage": "Texto...", "question": "É falso?", "answer": False},
        ]

    def test_bool_label_passed_through(self, monkeypatch):
        task = BoolQPT()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [{"passage": "Texto...", "question": "?", "label": True}],
        )
        examples = task.load_data({})
        assert examples[0]["answer"] is True


class TestBRoverbsLoadData:
    def test_correct_index_converted_to_letter(self, monkeypatch):
        task = BRoverbsTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "input": "Água mole em pedra dura...",
                    "alternatives": ["tanto bate até que fura", "não faz mal a ninguém"],
                    "correct_index": 0,
                }
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {
                "question": "Água mole em pedra dura...",
                "options": ["tanto bate até que fura", "não faz mal a ninguém"],
                "answer": "A",
            }
        ]


class TestCopaPTLoadData:
    def test_cause_question_uses_porque_connector(self, monkeypatch):
        task = CopaPTTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "premise": "O homem caiu",
                    "choice1": "escorregou",
                    "choice2": "correu",
                    "question": "cause",
                    "label": 0,
                }
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {
                "question": "O homem caiu porque...",
                "options": ["escorregou", "correu"],
                "answer": "A",
            }
        ]

    def test_effect_question_uses_portanto_connector(self, monkeypatch):
        task = CopaPTTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "premise": "Choveu",
                    "choice1": "a rua secou",
                    "choice2": "a rua alagou",
                    "question": "effect",
                    "label": 1,
                }
            ],
        )
        examples = task.load_data({})
        assert examples[0]["question"] == "Choveu portanto..."
        assert examples[0]["answer"] == "B"


class TestEnemLoadData:
    def test_real_schema_fields_mapped(self, monkeypatch):
        task = EnemTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "question": "2 + 2 = ?",
                    "alternatives": ["3", "4", "5", "6", "7"],
                    "label": "B",
                }
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"question": "2 + 2 = ?", "options": ["3", "4", "5", "6", "7"], "answer": "B"}
        ]


class TestHateBRLoadData:
    def test_offensive_language_bool_mapped_to_label(self, monkeypatch):
        task = HateBRTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {"instagram_comments": "comentario ruim", "offensive_language": True},
                {"instagram_comments": "comentario bom", "offensive_language": False},
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"text": "comentario ruim", "label": "odio"},
            {"text": "comentario bom", "label": "nao_odio"},
        ]


class TestMRPCPTLoadData:
    def test_int_label_mapped_to_sim_nao(self, monkeypatch):
        task = MRPCPTTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {"sentence1": "A", "sentence2": "B", "label": 1},
                {"sentence1": "C", "sentence2": "D", "label": 0},
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"sentence1": "A", "sentence2": "B", "label": "sim"},
            {"sentence1": "C", "sentence2": "D", "label": "nao"},
        ]


class TestOABBenchLoadData:
    def test_choices_dict_flattened(self, monkeypatch):
        task = OABBenchTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "question": "Segundo o CPC...",
                    "choices": {"text": ["opt B", "opt A"], "label": ["B", "A"]},
                    "answerKey": "A",
                }
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"question": "Segundo o CPC...", "options": ["opt A", "opt B"], "answer": "A"}
        ]


class TestPublicHearingBRLoadData:
    def test_nested_metadados_assunto_extracted(self, monkeypatch):
        task = PublicHearingBR()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {
                    "transcricao": "Texto da audiência...",
                    "metadados": {"assunto": "Saúde"},
                    "materia": "Resumo da matéria",
                }
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"text": "Texto da audiência...", "label": "saúde", "summary": "Resumo da matéria"}
        ]

    def test_missing_metadados_falls_back_to_label(self, monkeypatch):
        task = PublicHearingBR()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [{"transcricao": "Texto", "label": "Economia"}],
        )
        examples = task.load_data({})
        assert examples[0]["label"] == "economia"


class TestRTEPTLoadData:
    def test_int_label_converted_entailment_semantics(self, monkeypatch):
        task = RTEPTTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {"premise": "A", "hypothesis": "B", "label": 0},
                {"premise": "C", "hypothesis": "D", "label": 1},
            ],
        )
        examples = task.load_data({})
        assert examples == [
            {"premise": "A", "hypothesis": "B", "label": "entailment"},
            {"premise": "C", "hypothesis": "D", "label": "not_entailment"},
        ]


class TestTweetSentBRLoadData:
    def test_string_label_mapped_case_insensitively(self, monkeypatch):
        task = TweetSentBRTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [
                {"sentence": "Que dia otimo!", "label": "Positive"},
                {"sentence": "Que dia ruim!", "label": "Negative"},
                {"sentence": "Dia comum", "label": "Neutral"},
            ],
        )
        examples = task.load_data({})
        assert [e["label"] for e in examples] == ["positivo", "negativo", "neutro"]

    def test_unknown_label_passed_through_raw(self, monkeypatch):
        task = TweetSentBRTask()
        monkeypatch.setattr(
            task,
            "_load_from_hub",
            lambda *a, **kw: [{"sentence": "texto", "label": "Mixed"}],
        )
        examples = task.load_data({})
        assert examples[0]["label"] == "Mixed"


class TestLegalBenchBRLoadData:
    """Uses `datasets.load_dataset` directly (not BaseTask._load_from_hub),
    so mock at that level instead."""

    class _FakeLabelFeature:
        names = ["no", "partial", "yes"]

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows
            self.features = {"label": TestLegalBenchBRLoadData._FakeLabelFeature()}

        def __iter__(self):
            return iter(self._rows)

    def test_class_label_int_resolved_via_features(self, monkeypatch):
        task = LegalBenchBR()
        fake_ds = self._FakeDataset([{"sentence": "O pedido foi negado", "label": 0}])
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: fake_ds)

        examples = task.load_data({})

        assert len(examples) == 1
        assert examples[0]["question"] == "O pedido foi negado"
        assert examples[0]["answer"] == "A"  # "no" is index 0 in _LABEL_ORDER
        assert examples[0]["options"] == [
            "O pedido foi negado (improcedente).",
            "O pedido foi parcialmente provido.",
            "O pedido foi provido (procedente).",
        ]


class TestLeNERBrLoadData:
    """Uses `datasets.load_dataset` directly; ner_tags is a
    Sequence[ClassLabel] resolved dynamically from ds.features."""

    class _FakeDataset:
        def __init__(self, rows, tag_names):
            self._rows = rows

            class _Feature:
                pass

            feature = _Feature()
            feature.names = tag_names
            wrapper = _Feature()
            wrapper.feature = feature
            self.features = {"ner_tags": wrapper}

        def __iter__(self):
            return iter(self._rows)

    def test_ner_tags_resolved_from_int_via_features(self, monkeypatch):
        task = LeNERBr()
        tag_names = ["O", "B-PESSOA", "I-PESSOA"]
        fake_ds = self._FakeDataset(
            [{"id": 1, "tokens": ["João", "foi", "ao", "mercado"], "ner_tags": [1, 2, 0, 0]}],
            tag_names,
        )
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **kw: fake_ds)

        examples = task.load_data({})

        assert examples == [
            {
                "id": 1,
                "tokens": ["João", "foi", "ao", "mercado"],
                "ner_tags": ["B-PESSOA", "I-PESSOA", "O", "O"],
                "text": "João foi ao mercado",
            }
        ]
