"""Tests for instruction data builder and chat template formatting.

Imports the real `format_gemma4_chat` and `InstructionDataBuilder._mask_prompt_tokens`
from src/data/instruction_data_builder.py rather than reimplementing them, so these
tests actually exercise production code (see tests/test_data_pipeline_fixes.py for the
prior fix that established this pattern — this file previously tested a fabricated
Gemma 2/3-style stand-in and gave zero regression protection for the real template).
"""

from src.data.instruction_data_builder import (
    GEMMA4_MODEL_PREFIX,
    GEMMA4_MODEL_SUFFIX,
    GEMMA4_USER_PREFIX,
    GEMMA4_USER_SUFFIX,
    InstructionDataBuilder,
    format_gemma4_chat,
)


class TestFormatGemma4Chat:
    """Test Gemma 4 chat template formatting."""

    def test_single_turn(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there!"},
        ]
        result = format_gemma4_chat(messages)
        assert "<|turn>user\nHello<turn|>" in result
        assert "<|turn>model\nHi there!<turn|>" in result

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "model", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "model", "content": "6"},
        ]
        result = format_gemma4_chat(messages)
        assert result.count("<|turn>user") == 2
        assert result.count("<|turn>model") == 2
        assert result.count("<turn|>") == 4

    def test_generation_prompt_no_think(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = format_gemma4_chat(messages, add_generation_prompt=True)
        assert result.endswith("<|turn>model\n")
        assert "<|channel>thought" not in result

    def test_generation_prompt_with_think(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = format_gemma4_chat(messages, add_generation_prompt=True, use_think=True)
        assert result.endswith("<|turn>model\n<|channel>thought\n")

    def test_think_mode_wraps_response(self):
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "model", "content": "thinking... answer"},
        ]
        result = format_gemma4_chat(messages, use_think=True)
        assert "<|channel>thought\nthinking... answer\n<channel|>" in result

    def test_assistant_role_accepted(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = format_gemma4_chat(messages)
        assert "<|turn>model\nHello<turn|>" in result

    def test_no_duplicate_bos(self):
        """format_gemma4_chat must NOT prepend a literal BOS string — the
        tokenizer adds it via add_special_tokens=True. A literal "<bos>"
        here would double it up at encode time."""
        result = format_gemma4_chat([{"role": "user", "content": "test"}])
        assert "<bos>" not in result

    def test_empty_messages(self):
        result = format_gemma4_chat([])
        assert result == ""

    def test_preserves_special_chars_in_content(self):
        messages = [
            {"role": "user", "content": "x = 1\ny = 2\nprint(x+y)"},
            {"role": "model", "content": "```python\n3\n```"},
        ]
        result = format_gemma4_chat(messages)
        assert "x = 1\ny = 2" in result
        assert "```python" in result

    def test_no_previous_think_in_multiturn(self):
        """Generation-prompt suffix starts a fresh, unclosed thought block —
        it must not include or leak any earlier turn's content."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "model", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = format_gemma4_chat(messages, add_generation_prompt=True, use_think=True)
        assert result.endswith("<|turn>model\n<|channel>thought\n")


class TestChatTemplateConstants:
    """Test that template constants are correct (Gemma 4's dedicated special
    tokens, NOT Gemma 2/3's <start_of_turn>/<end_of_turn> plain-text markers)."""

    def test_user_prefix(self):
        assert GEMMA4_USER_PREFIX == "<|turn>user\n"

    def test_user_suffix(self):
        assert GEMMA4_USER_SUFFIX == "<turn|>\n"

    def test_model_prefix(self):
        assert GEMMA4_MODEL_PREFIX == "<|turn>model\n"

    def test_model_suffix(self):
        assert GEMMA4_MODEL_SUFFIX == "<turn|>\n"

    def test_no_trailing_spaces(self):
        for token in [
            GEMMA4_USER_PREFIX,
            GEMMA4_USER_SUFFIX,
            GEMMA4_MODEL_PREFIX,
            GEMMA4_MODEL_SUFFIX,
        ]:
            assert not token.endswith(" ")


class TestMaskPromptTokens:
    """Test InstructionDataBuilder._mask_prompt_tokens — the real completions-
    only label-masking logic (used by cpt/sft training, not a reimplementation).
    The method doesn't touch `self`, so it can be called unbound on `None`.
    """

    def _mask(self, input_ids, response_template_ids, turn_start_ids=None):
        labels = list(input_ids)
        return InstructionDataBuilder._mask_prompt_tokens(
            None, input_ids, labels, response_template_ids, turn_start_ids
        )

    def test_mask_prompt_tokens(self):
        input_ids = [1, 2, 3, 100, 200, 300, 4, 5, 6]
        response_template_ids = [100, 200, 300]

        masked_labels = self._mask(input_ids, response_template_ids)

        assert masked_labels[:6] == [-100] * 6
        assert masked_labels[6:] == [4, 5, 6]

    def test_multi_turn_unmask_stops_at_next_turn(self):
        """A multi-turn sequence must not leak the following user turn into
        the unmasked (trained-on) region for an earlier model response."""
        turn_start_ids = [9]
        response_template_ids = [9, 100]  # "<|turn>" + "model"
        # turn: <|turn>model A1 <|turn>user Q2 <|turn>model A2
        input_ids = [9, 100, 11, 9, 200, 12, 9, 100, 13]
        labels = list(input_ids)

        masked = InstructionDataBuilder._mask_prompt_tokens(
            None, input_ids, labels, response_template_ids, turn_start_ids
        )

        # First model response (index 2) unmasked, stops before the next <|turn> at index 3
        assert masked[2] == 11
        assert masked[3] == -100  # next turn's <|turn> marker stays masked
        assert masked[4] == -100  # "user" role token, masked
        # Second model response (index 8) unmasked, runs to end of sequence
        assert masked[8] == 13

    def test_missing_template_falls_back_to_unmasked(self):
        """If the response template never appears, masking is skipped
        entirely (logged as a warning) rather than silently training on
        garbage positions."""
        input_ids = [1, 2, 3, 4, 5]
        response_template_ids = [999]

        masked = self._mask(input_ids, response_template_ids)

        assert masked == input_ids
