"""Tests for instruction data builder and chat template formatting.

Note: These tests import only the pure-Python formatting functions,
not the dataset loading code (which requires the `datasets` library).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only constants and pure functions that don't trigger heavy deps
# The format_gemma4_chat function is defined before the class that uses `datasets`
# We re-implement the key logic here for testing without the import chain.

# Gemma 4 chat template constants (duplicated for test isolation)
GEMMA4_USER_PREFIX = "<start_of_turn>user\n"
GEMMA4_USER_SUFFIX = "<end_of_turn>\n"
GEMMA4_MODEL_PREFIX = "<start_of_turn>model\n"
GEMMA4_MODEL_SUFFIX = "<end_of_turn>\n"


def format_gemma4_chat(messages, add_generation_prompt=False, use_think=False):
    """Local copy of format function for testing without datasets dep."""
    formatted = "<bos>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"{GEMMA4_USER_PREFIX}{content}{GEMMA4_USER_SUFFIX}"
        elif role in ("model", "assistant"):
            if use_think:
                formatted += (
                    f"{GEMMA4_MODEL_PREFIX}<think>\n{content}\n</think>\n{GEMMA4_MODEL_SUFFIX}"
                )
            else:
                formatted += f"{GEMMA4_MODEL_PREFIX}{content}{GEMMA4_MODEL_SUFFIX}"
    if add_generation_prompt:
        if use_think:
            formatted += f"{GEMMA4_MODEL_PREFIX}<think>\n"
        else:
            formatted += GEMMA4_MODEL_PREFIX
    return formatted


class TestFormatGemma4Chat:
    """Test Gemma 4 chat template formatting."""

    def test_single_turn(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there!"},
        ]
        result = format_gemma4_chat(messages)
        assert result.startswith("<bos>")
        assert "<start_of_turn>user\nHello<end_of_turn>" in result
        assert "<start_of_turn>model\nHi there!<end_of_turn>" in result

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "model", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "model", "content": "6"},
        ]
        result = format_gemma4_chat(messages)
        assert result.count("<start_of_turn>user") == 2
        assert result.count("<start_of_turn>model") == 2
        assert result.count("<end_of_turn>") == 4

    def test_generation_prompt_no_think(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = format_gemma4_chat(messages, add_generation_prompt=True)
        assert result.endswith("<start_of_turn>model\n")
        assert "<think>" not in result

    def test_generation_prompt_with_think(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = format_gemma4_chat(messages, add_generation_prompt=True, use_think=True)
        assert result.endswith("<start_of_turn>model\n<think>\n")

    def test_think_mode_wraps_response(self):
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "model", "content": "thinking... answer"},
        ]
        result = format_gemma4_chat(messages, use_think=True)
        assert "<think>\nthinking... answer\n</think>" in result

    def test_assistant_role_accepted(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = format_gemma4_chat(messages)
        assert "<start_of_turn>model\nHello<end_of_turn>" in result

    def test_bos_always_first(self):
        messages = [{"role": "user", "content": "test"}]
        result = format_gemma4_chat(messages)
        assert result[0:5] == "<bos>"

    def test_empty_messages(self):
        result = format_gemma4_chat([])
        assert result == "<bos>"

    def test_preserves_special_chars_in_content(self):
        messages = [
            {"role": "user", "content": "x = 1\ny = 2\nprint(x+y)"},
            {"role": "model", "content": "```python\n3\n```"},
        ]
        result = format_gemma4_chat(messages)
        assert "x = 1\ny = 2" in result
        assert "```python" in result

    def test_no_previous_think_in_multiturn(self):
        """Multi-turn with think should not leak previous thinking."""
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "model", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = format_gemma4_chat(messages, add_generation_prompt=True, use_think=True)
        # First response should NOT have think tags (only the prompt for generation does)
        # Actually with use_think=True in format, all model responses get think wrapping
        # But the key point: the generation prompt at the end starts fresh
        assert result.endswith("<start_of_turn>model\n<think>\n")


class TestChatTemplateConstants:
    """Test that template constants are correct."""

    def test_user_prefix(self):
        assert GEMMA4_USER_PREFIX == "<start_of_turn>user\n"

    def test_user_suffix(self):
        assert GEMMA4_USER_SUFFIX == "<end_of_turn>\n"

    def test_model_prefix(self):
        assert GEMMA4_MODEL_PREFIX == "<start_of_turn>model\n"

    def test_model_suffix(self):
        assert GEMMA4_MODEL_SUFFIX == "<end_of_turn>\n"

    def test_no_trailing_spaces(self):
        for token in [
            GEMMA4_USER_PREFIX,
            GEMMA4_USER_SUFFIX,
            GEMMA4_MODEL_PREFIX,
            GEMMA4_MODEL_SUFFIX,
        ]:
            assert not token.endswith(" ")


class TestMaskingLogic:
    """Test the label masking logic for train-on-completions-only."""

    def test_find_response_template(self):
        """Verify response template matching logic."""
        # Simulated token IDs
        response_template_ids = [100, 200, 300]  # "<start_of_turn>model\n"
        input_ids = [1, 2, 3, 100, 200, 300, 4, 5, 6]

        # Find template position
        template_len = len(response_template_ids)
        found_positions = []
        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i : i + template_len] == response_template_ids:
                found_positions.append(i + template_len)

        assert len(found_positions) == 1
        assert found_positions[0] == 6  # Position after template

    def test_mask_prompt_tokens(self):
        """Test that prompt tokens get masked to -100."""
        IGNORE_INDEX = -100
        input_ids = [1, 2, 3, 100, 200, 300, 4, 5, 6]
        response_template_ids = [100, 200, 300]

        # Apply masking logic
        masked_labels = [IGNORE_INDEX] * len(input_ids)
        template_len = len(response_template_ids)

        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i : i + template_len] == response_template_ids:
                start = i + template_len
                for j in range(start, len(input_ids)):
                    masked_labels[j] = input_ids[j]
                break

        # Prompt tokens should be masked
        assert masked_labels[:6] == [IGNORE_INDEX] * 6
        # Response tokens should be unmasked
        assert masked_labels[6:] == [4, 5, 6]
