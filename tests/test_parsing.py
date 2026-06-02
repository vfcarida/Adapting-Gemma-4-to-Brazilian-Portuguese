"""Tests for prediction parsing in evaluation tasks."""

from src.eval.tasks.assin2_rte import Assin2RTETask
from src.eval.tasks.copa_pt import CopaPTTask
from src.eval.tasks.enem import EnemTask
from src.eval.tasks.hatebr import HateBRTask
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
    def setup_method(self):
        self.task = CopaPTTask()

    def test_choice_1(self):
        assert self.task.parse_prediction("1") == "1"
        assert self.task.parse_prediction("1. primeira opcao") == "1"

    def test_choice_2(self):
        assert self.task.parse_prediction("2") == "2"


class TestMRPCParsing:
    def setup_method(self):
        self.task = MRPCPTTask()

    def test_yes(self):
        assert self.task.parse_prediction("sim") == "sim"
        assert self.task.parse_prediction("Sim, sao parafrases") == "sim"

    def test_no(self):
        assert self.task.parse_prediction("nao") == "nao"
        assert self.task.parse_prediction("Diferentes") == "nao"
