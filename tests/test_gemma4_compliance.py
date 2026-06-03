"""Testes de conformidade Gemma 4 — valida camada de prompts.

Cobre:
- Prompt sem thinking
- Prompt com thinking
- Pensamento em multi-turn
- Stripping correto do histórico
- Compatibilidade de parsing de respostas
- Diferenciação text-only/IT/thinking/multimodal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prompt_builders import (
    GEMMA4_END_OF_TURN,
    GEMMA4_START_OF_TURN,
    GEMMA4_THINK_CLOSE,
    GEMMA4_THINK_OPEN,
    BaselinePromptBuilder,
    Gemma4PromptBuilder,
)
from src.eval.prompt_templates import (
    PromptBuilder,
    _wrap_gemma4_legacy,
    extract_thought,
    strip_thought,
)

# =============================================================================
# Fixtures
# =============================================================================


class MockTokenizerNoChat:
    """Tokenizer sem chat template."""

    pass


class MockTokenizerWithChat:
    """Tokenizer com apply_chat_template funcional."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")
        return "".join(parts)


# =============================================================================
# Testes de formato sem thinking
# =============================================================================


class TestNoThinking:
    """Prompts sem modo de pensamento (modo padrão)."""

    def test_single_user_message(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [{"role": "user", "content": "Olá"}]
        result = builder.format_for_inference(messages, think_mode="off")

        assert f"{GEMMA4_START_OF_TURN}user\nOlá{GEMMA4_END_OF_TURN}" in result
        assert result.endswith(f"{GEMMA4_START_OF_TURN}model\n")
        assert GEMMA4_THINK_OPEN not in result

    def test_system_plus_user(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [
            {"role": "system", "content": "Você é um assistente."},
            {"role": "user", "content": "Pergunta"},
        ]
        result = builder.format_for_inference(messages, think_mode="off")

        assert f"{GEMMA4_START_OF_TURN}system\nVocê é um assistente." in result
        assert f"{GEMMA4_START_OF_TURN}user\nPergunta" in result
        assert GEMMA4_THINK_OPEN not in result

    def test_with_chat_template_tokenizer(self):
        builder = Gemma4PromptBuilder(MockTokenizerWithChat())
        messages = [{"role": "user", "content": "Test"}]
        result = builder.format_for_inference(messages, think_mode="off")

        # Should use apply_chat_template output
        assert "Test" in result
        assert "<start_of_turn>model\n" in result
        assert GEMMA4_THINK_OPEN not in result

    def test_training_format_includes_model_response(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [
            {"role": "user", "content": "Pergunta"},
            {"role": "model", "content": "Resposta"},
        ]
        result = builder.format_for_training(messages, think_mode="off")

        assert "Pergunta" in result
        assert "Resposta" in result
        # Training should not add generation prompt
        assert not result.endswith(f"{GEMMA4_START_OF_TURN}model\n")


# =============================================================================
# Testes com thinking
# =============================================================================


class TestWithThinking:
    """Prompts com modo de pensamento ativo."""

    def test_think_on_inference(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [{"role": "user", "content": "Resolva 2+2"}]
        result = builder.format_for_inference(messages, think_mode="on")

        # Should end with model prefix + think open
        assert result.endswith(f"{GEMMA4_START_OF_TURN}model\n{GEMMA4_THINK_OPEN}\n")

    def test_think_budget_inference(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [{"role": "user", "content": "Pergunta"}]
        result = builder.format_for_inference(messages, think_mode="budget")

        # Should have empty think channel
        assert f"{GEMMA4_THINK_OPEN}\n{GEMMA4_THINK_CLOSE}\n" in result

    def test_think_on_training(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [
            {"role": "user", "content": "Pergunta"},
            {"role": "model", "content": "Resposta"},
        ]
        result = builder.format_for_training(messages, think_mode="on")

        # Model response should be wrapped with think
        assert GEMMA4_THINK_OPEN in result

    def test_think_off_explicit(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [{"role": "user", "content": "Test"}]
        result = builder.format_for_inference(messages, think_mode="off")

        assert GEMMA4_THINK_OPEN not in result
        assert GEMMA4_THINK_CLOSE not in result


# =============================================================================
# Testes multi-turn com pensamento
# =============================================================================


class TestMultiTurnThinking:
    """Multi-turn: pensamentos devem ser removidos do histórico."""

    def test_strip_thought_from_previous_turn(self):
        """Ao construir multi-turn, pensamentos de turnos anteriores devem ser removidos."""
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())

        # Simula output do modelo com pensamento
        model_output_with_think = "<think>Preciso calcular...</think>A resposta é 4."

        # Strip para construir próximo turno
        cleaned = builder.strip_thought(model_output_with_think)
        assert "Preciso calcular" not in cleaned
        assert "A resposta é 4." in cleaned

        # Construir próximo turno com histórico limpo
        messages = [
            {"role": "user", "content": "Quanto é 2+2?"},
            {"role": "model", "content": cleaned},  # Sem pensamento
            {"role": "user", "content": "E 3+3?"},
        ]
        result = builder.format_for_inference(messages, think_mode="on")

        # Verificar que o histórico não tem pensamento antigo
        assert "Preciso calcular" not in result
        # Mas o novo turno termina com <think> para gerar pensamento
        assert result.endswith(f"{GEMMA4_THINK_OPEN}\n")

    def test_extract_thought_and_answer(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())

        text = "<think>Passo 1: 2+2=4\nPasso 2: confirmar</think>A resposta é 4."
        thought, answer = builder.extract_thought(text)

        assert "Passo 1" in thought
        assert "Passo 2" in thought
        assert "A resposta é 4." in answer
        assert "<think>" not in answer

    def test_multiple_think_blocks(self):
        """Múltiplos blocos de pensamento (edge case)."""
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())

        text = "<think>primeiro</think>resposta1<think>segundo</think>resposta2"
        thought, answer = builder.extract_thought(text)

        assert "primeiro" in thought
        assert "segundo" in thought
        assert "resposta1" in answer
        assert "resposta2" in answer

    def test_empty_think_block(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())

        text = "<think></think>Resposta direta"
        cleaned = builder.strip_thought(text)
        assert cleaned == "Resposta direta"

    def test_no_think_block(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())

        text = "Resposta sem pensamento"
        thought, answer = builder.extract_thought(text)
        assert thought == ""
        assert answer == "Resposta sem pensamento"


# =============================================================================
# Testes de parsing de respostas
# =============================================================================


class TestResponseParsing:
    """Parsing de respostas de modelos em benchmarks."""

    def test_strip_thought_before_parsing(self):
        """O pipeline de eval deve strip thought antes de parse."""
        # Simula output bruto do modelo
        raw = "<think>Vou pensar... A opção C parece correta</think>C"
        cleaned = strip_thought(raw)
        assert cleaned == "C"

    def test_strip_multiline_thought(self):
        raw = """<think>
Analisando as opções:
- A: incorreta
- B: incorreta
- C: correta porque Brasília é a capital
</think>
C"""
        cleaned = strip_thought(raw)
        assert cleaned.strip() == "C"

    def test_extract_preserves_answer_formatting(self):
        raw = "<think>raciocínio</think>A resposta é: **B**"
        thought, answer = extract_thought(raw)
        assert "raciocínio" in thought
        assert "A resposta é: **B**" in answer

    def test_strip_handles_unicode(self):
        raw = "<think>análise com acentuação</think>Ação"
        cleaned = strip_thought(raw)
        assert cleaned == "Ação"


# =============================================================================
# Testes de compatibilidade text-only / IT
# =============================================================================


class TestModelModes:
    """Diferenciação entre modos de modelo."""

    def test_baseline_non_chat(self):
        """Modelos baseline (Sabia-7B) não usam chat template."""
        builder = BaselinePromptBuilder(
            prompt_prefix="",
            user_prefix="Pergunta: ",
            model_prefix="Resposta: ",
        )
        messages = [{"role": "user", "content": "Teste"}]
        result = builder.format_for_inference(messages)

        assert "Pergunta: Teste" in result
        assert "Resposta: " in result
        assert "<start_of_turn>" not in result

    def test_eval_prompt_builder_non_chat(self):
        """PromptBuilder (eval) em modo não-chat."""
        pb = PromptBuilder(tokenizer=None, is_chat_model=False)
        result = pb.format_prompt(None, "Teste")
        assert result == "Teste"
        assert "<start_of_turn>" not in result

    def test_eval_prompt_builder_chat_fallback(self):
        """PromptBuilder (eval) em modo chat sem tokenizer."""
        pb = PromptBuilder(tokenizer=None, is_chat_model=True)
        result = pb.format_prompt(None, "Teste", think_mode="off")
        assert "<start_of_turn>user" in result
        assert "<start_of_turn>model" in result

    def test_legacy_wrap_consistency(self):
        """_wrap_gemma4_legacy deve ser consistente com Gemma4PromptBuilder."""
        legacy = _wrap_gemma4_legacy("Teste", think_mode="off")
        assert "<bos>" in legacy
        assert "<start_of_turn>user\nTeste" in legacy


# =============================================================================
# Testes de few-shot
# =============================================================================


class TestFewShot:
    """Few-shot prompt construction."""

    def test_baseline_few_shot(self):
        builder = BaselinePromptBuilder(
            user_prefix="Q: ",
            model_prefix="A: ",
            model_suffix="\n\n",
        )
        messages = [{"role": "user", "content": "Nova pergunta"}]
        few_shot = [
            {"role": "user", "content": "Exemplo 1"},
            {"role": "model", "content": "Resposta 1"},
        ]
        result = builder.format_for_inference(messages, few_shot_examples=few_shot)

        assert "Q: Exemplo 1" in result
        assert "A: Resposta 1" in result
        assert "Q: Nova pergunta" in result
        # Few shot comes before the query
        assert result.index("Exemplo 1") < result.index("Nova pergunta")

    def test_task_template_few_shot(self):
        from src.eval.prompt_templates import TaskPromptTemplate

        examples = [
            {"question": "1+1?", "options": ["1", "2"], "answer": "B"},
        ]
        template = TaskPromptTemplate("enem", num_shots=1, examples=examples)
        test_ex = {"question": "2+2?", "options": ["3", "4"], "answer": "B"}
        prompt = template.format_prompt(test_ex, think_mode="off")

        # Should include the few-shot example with answer
        assert "Resposta: B" in prompt
        # Test example should not have answer revealed
        assert prompt.count("Resposta: B") == 1


# =============================================================================
# Testes de edge cases
# =============================================================================


class TestEdgeCases:
    """Edge cases e robustez."""

    def test_empty_content(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [{"role": "user", "content": ""}]
        result = builder.format_for_inference(messages)
        # Should not crash, just format empty content
        assert f"{GEMMA4_START_OF_TURN}user" in result

    def test_very_long_content(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        long_text = "x" * 10000
        messages = [{"role": "user", "content": long_text}]
        result = builder.format_for_inference(messages)
        assert long_text in result

    def test_special_characters_in_content(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        special = "Texto com <tags> e &entidades; e 'aspas' e \"duplas\""
        messages = [{"role": "user", "content": special}]
        result = builder.format_for_inference(messages)
        assert special in result

    def test_think_tags_in_user_message(self):
        """Se usuário envia <think> no texto, não deve confundir o parser."""
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())

        # User message contains think-like text
        user_msg = "O que significa a tag <think> no Gemma?"
        messages = [{"role": "user", "content": user_msg}]
        result = builder.format_for_inference(messages, think_mode="off")
        # The <think> in user message is just text, not a mode indicator
        assert user_msg in result

    def test_unknown_think_mode_noop(self):
        builder = Gemma4PromptBuilder(MockTokenizerNoChat())
        messages = [{"role": "user", "content": "Test"}]
        result = builder.format_for_inference(messages, think_mode="invalid_mode")
        # Should not crash, just return without think modifications
        assert "Test" in result
