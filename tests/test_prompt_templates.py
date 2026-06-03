"""Testes para formatação de templates de prompt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.prompt_templates import (
    TASK_FORMATTERS,
    TASK_INSTRUCTIONS,
    PromptBuilder,
    TaskPromptTemplate,
    _wrap_gemma4_legacy,
    extract_thought,
    get_prompt_template,
    strip_thought,
)


class TestGemma4Wrapping:
    """Testa formatação do chat template Gemma 4."""

    def test_basic_wrap_think_off(self):
        result = _wrap_gemma4_legacy("Hello", think_mode="off")
        assert result.startswith("<bos><start_of_turn>user\n")
        assert "Hello" in result
        assert result.endswith("<end_of_turn>\n<start_of_turn>model\n")
        assert "<think>" not in result

    def test_basic_wrap_think_on(self):
        result = _wrap_gemma4_legacy("Hello", think_mode="on")
        assert result.endswith("<start_of_turn>model\n<think>\n")
        assert "<think>" in result

    def test_empty_channel(self):
        result = _wrap_gemma4_legacy("Hello", think_mode="empty_channel")
        assert "<think>\n</think>\n" in result

    def test_preserves_content(self):
        content = "Qual é a capital do Brasil?"
        result = _wrap_gemma4_legacy(content)
        assert content in result

    def test_bos_token_present(self):
        result = _wrap_gemma4_legacy("test")
        assert result.startswith("<bos>")


class TestStripThought:
    """Testa remoção de blocos de pensamento."""

    def test_strip_simple(self):
        text = "<think>pensando...</think>A resposta é B"
        assert strip_thought(text) == "A resposta é B"

    def test_strip_multiline(self):
        text = "<think>\nlinha1\nlinha2\n</think>\nResposta: C"
        result = strip_thought(text)
        assert "Resposta: C" in result
        assert "<think>" not in result

    def test_no_thought_block(self):
        text = "Resposta direta sem pensamento"
        assert strip_thought(text) == text

    def test_extract_thought_and_answer(self):
        text = "<think>Preciso calcular 2+2=4</think>A resposta é 4"
        thought, answer = extract_thought(text)
        assert "calcular" in thought
        assert "resposta é 4" in answer


class TestPromptBuilder:
    """Testa o PromptBuilder com e sem chat template."""

    def test_non_chat_model_returns_user_msg(self):
        builder = PromptBuilder(tokenizer=None, is_chat_model=False)
        result = builder.format_prompt(None, "Pergunta teste")
        assert result == "Pergunta teste"

    def test_chat_model_fallback_gemma4(self):
        builder = PromptBuilder(tokenizer=None, is_chat_model=True)
        result = builder.format_prompt(None, "Pergunta teste", think_mode="off")
        assert "<start_of_turn>user" in result
        assert "Pergunta teste" in result
        assert "<start_of_turn>model" in result

    def test_chat_model_with_system(self):
        builder = PromptBuilder(tokenizer=None, is_chat_model=True)
        result = builder.format_prompt("Você é um assistente.", "Olá", think_mode="off")
        assert "system" in result
        assert "Você é um assistente." in result

    def test_think_on_adds_think_tag(self):
        builder = PromptBuilder(tokenizer=None, is_chat_model=True)
        result = builder.format_prompt(None, "test", think_mode="on")
        assert result.endswith("<think>\n")


class TestTaskPromptTemplate:
    """Testa template de prompt por tarefa."""

    def test_enem_format(self):
        template = get_prompt_template("enem", num_shots=0)
        example = {
            "question": "Qual a capital do Brasil?",
            "options": ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Curitiba"],
            "answer": "C",
        }
        prompt = template.format_prompt(example, think_mode="off")
        assert "Qual a capital do Brasil?" in prompt
        assert "A) São Paulo" in prompt
        assert "C) Brasília" in prompt
        assert "Resposta:" in prompt

    def test_rte_format(self):
        template = get_prompt_template("assin2_rte", num_shots=0)
        example = {
            "premise": "O gato está no telhado",
            "hypothesis": "Há um animal no telhado",
            "label": "entailment",
        }
        prompt = template.format_prompt(example)
        assert "Premissa:" in prompt
        assert "Hipótese:" in prompt

    def test_classification_format(self):
        template = get_prompt_template("hatebr", num_shots=0)
        example = {"text": "Texto de exemplo", "label": "nao_odio"}
        prompt = template.format_prompt(example)
        assert "Texto: Texto de exemplo" in prompt

    def test_think_on_format(self):
        template = get_prompt_template("enem", num_shots=0)
        example = {"question": "Test?", "options": ["A", "B"], "answer": "A"}
        prompt = template.format_prompt(example, think_mode="on")
        assert "<think>" in prompt

    def test_all_tasks_have_instructions(self):
        expected_tasks = [
            "enem",
            "bluex",
            "assin2_rte",
            "assin2_sts",
            "copa_pt",
            "boolq_pt",
            "mrpc_pt",
            "rte_pt",
            "hatebr",
            "tweet_sentbr",
            "oab_bench",
            "broverbs",
            "capitu",
            "math_pt",
            "lener_br",
            "legalbench_br",
            "publichearing_br",
            "donotanswer_pt",
            "xlsum_pt",
            "mmlu_en",
            "hellaswag_en",
            "arc_en",
        ]
        for task in expected_tasks:
            assert task in TASK_INSTRUCTIONS, f"Faltando instrução para {task}"

    def test_all_tasks_have_formatters(self):
        expected_tasks = [
            "enem",
            "bluex",
            "assin2_rte",
            "assin2_sts",
            "copa_pt",
            "boolq_pt",
            "mrpc_pt",
            "rte_pt",
            "hatebr",
            "tweet_sentbr",
            "oab_bench",
            "broverbs",
            "capitu",
            "math_pt",
            "lener_br",
            "legalbench_br",
            "publichearing_br",
            "donotanswer_pt",
            "xlsum_pt",
            "mmlu_en",
            "hellaswag_en",
            "arc_en",
        ]
        for task in expected_tasks:
            assert task in TASK_FORMATTERS, f"Faltando formatador para {task}"


class TestFewShot:
    """Testa construção de prompts few-shot."""

    def test_zero_shot_no_examples(self):
        template = TaskPromptTemplate("enem", num_shots=0)
        example = {"question": "Q?", "options": ["A", "B"], "answer": "A"}
        prompt = template.format_prompt(example)
        assert "Resposta: A" not in prompt
        assert "Resposta:" in prompt

    def test_few_shot_with_examples(self):
        few_shot_examples = [
            {"question": "1+1?", "options": ["1", "2", "3"], "answer": "B"},
        ]
        template = TaskPromptTemplate("enem", num_shots=1, examples=few_shot_examples)
        test_example = {"question": "2+2?", "options": ["3", "4", "5"], "answer": "B"}
        prompt = template.format_prompt(test_example)
        assert "Resposta: B" in prompt
        assert prompt.count("Resposta: B") == 1
