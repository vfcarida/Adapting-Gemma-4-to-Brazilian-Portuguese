"""Tests for prediction parsing in evaluation tasks."""

from src.eval.tasks.assin2_rte import Assin2RTETask
from src.eval.tasks.copa_pt import CopaPTTask
from src.eval.tasks.enem import EnemTask
from src.eval.tasks.hatebr import HateBRTask
from src.eval.tasks.math_pt import MathPT
from src.eval.tasks.mrpc_pt import MRPCPTTask
from src.eval.tasks.tweet_sentbr import TweetSentBRTask


class TestLetterExtraction:
    def setup_method(self):
        self.task = EnemTask()

    def test_single_letter(self):
        assert self.task.parse_prediction("A") == "A"
        assert self.task.parse_prediction("b") == "B"

    def test_letter_with_paren(self):
        assert self.task.parse_prediction("A)") == "A"
        assert self.task.parse_prediction("B) resposta") == "B"

    def test_letter_with_period(self):
        assert self.task.parse_prediction("C. a resposta") == "C"

    def test_letter_in_text(self):
        assert self.task.parse_prediction("A alternativa correta e A") == "A"

    def test_letter_with_whitespace(self):
        assert self.task.parse_prediction("  D  ") == "D"


class TestRTEParsing:
    def setup_method(self):
        self.task = Assin2RTETask()

    def test_entailment(self):
        assert self.task.parse_prediction("entailment") == "entailment"
        assert self.task.parse_prediction("Entailment.") == "entailment"

    def test_not_entailment(self):
        assert self.task.parse_prediction("not_entailment") == "not_entailment"
        assert self.task.parse_prediction("It is not entailment") == "not_entailment"


class TestHateBRParsing:
    def setup_method(self):
        self.task = HateBRTask()

    def test_hate(self):
        assert self.task.parse_prediction("odio") == "odio"
        assert self.task.parse_prediction("Discurso de odio") == "odio"

    def test_not_hate(self):
        assert self.task.parse_prediction("nao_odio") == "nao_odio"
        assert self.task.parse_prediction("Nao e odio") == "nao_odio"


class TestSentimentParsing:
    def setup_method(self):
        self.task = TweetSentBRTask()

    def test_positive(self):
        assert self.task.parse_prediction("positivo") == "positivo"
        assert self.task.parse_prediction("Positive sentiment") == "positivo"

    def test_negative(self):
        assert self.task.parse_prediction("negativo") == "negativo"

    def test_neutral(self):
        assert self.task.parse_prediction("neutro") == "neutro"
        assert self.task.parse_prediction("unknown") == "neutro"  # Default


class TestCopaParsing:
    """CoPA-PT now uses letter-based answers (A/B), matching the "A) / B)"
    options actually rendered by the default MC formatter and the
    letter-based TASK_INSTRUCTIONS["copa_pt"] wording — previously the
    parser looked for "1"/"2" while the rendered prompt showed "A)"/"B)",
    which pinned accuracy near chance."""

    def setup_method(self):
        self.task = CopaPTTask()

    def test_choice_a(self):
        assert self.task.parse_prediction("A") == "A"
        assert self.task.parse_prediction("A) primeira opcao") == "A"

    def test_choice_b(self):
        assert self.task.parse_prediction("B") == "B"


class TestMRPCParsing:
    def setup_method(self):
        self.task = MRPCPTTask()

    def test_yes(self):
        assert self.task.parse_prediction("sim") == "sim"
        assert self.task.parse_prediction("Sim, sao parafrases") == "sim"

    def test_no(self):
        assert self.task.parse_prediction("nao") == "nao"
        assert self.task.parse_prediction("Diferentes") == "nao"


class TestMathPTLoadData:
    """Math-PT (tiagoteixeira03/MATH-PT) was newly wired in with no prior
    coverage — its choices-dict schema (`{"A": ..., "E": ...}`, not the
    ARC/BLUEX/OAB-style `{"text": [...], "label": [...]}`) is untested
    elsewhere, so exercise `load_data`'s normalization directly against a
    mocked `_load_from_hub` (no network)."""

    def setup_method(self):
        self.task = MathPT()

    def test_choices_dict_ordered_by_key(self, monkeypatch):
        raw_item = {
            "problem": "Quanto é 2 + 2?",
            "choices": {"C": "5", "A": "3", "B": "4", "D": "6", "E": "7"},
            "answer": "B",
        }
        monkeypatch.setattr(self.task, "_load_from_hub", lambda *a, **kw: [raw_item])

        examples = self.task.load_data({})

        assert examples == [
            {
                "question": "Quanto é 2 + 2?",
                "options": ["3", "4", "5", "6", "7"],  # sorted A..E, not insertion order
                "answer": "B",
            }
        ]

    def test_max_samples_applied(self, monkeypatch):
        raw_items = [{"problem": f"P{i}", "choices": {"A": "x"}, "answer": "A"} for i in range(10)]
        monkeypatch.setattr(self.task, "_load_from_hub", lambda *a, **kw: raw_items)

        examples = self.task.load_data({"max_samples": 3})

        assert len(examples) == 3


class TestMathPTParsing:
    def setup_method(self):
        self.task = MathPT()

    def test_letter_preferred_over_number(self):
        assert self.task.parse_prediction("D") == "D"
        assert self.task.parse_prediction("A resposta é D) 16 cm") == "D"

    def test_leading_article_not_mistaken_for_letter_answer(self):
        """ "A resposta é 16" starts with the Portuguese article "A" — this
        used to be misread as answer letter "A" by _extract_letter's old
        overly-broad standalone-letter rule (see base_task.py fix). It
        should fall through to the numeric fallback instead."""
        assert self.task.parse_prediction("A resposta é 16") == "16"

    def test_gold_label_uppercased(self):
        assert self.task.get_gold_label({"answer": "b"}) == "B"
