import json

import pytest
from miniopd.evaluation.prompt import (
    SYSTEM_PROMPT,
    PromptTemplateError,
    build_prompt,
    build_sft_messages,
    render_generation_prompt,
    render_sft_example,
    save_rendered_prompt_debug,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self, conversation, *, tokenize: bool, add_generation_prompt: bool
    ):
        self.calls.append((conversation, tokenize, add_generation_prompt))
        if tokenize:
            return [10, 20, 30] if add_generation_prompt else [10, 20, 30, 40]
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in conversation
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")


def test_prompt_contract_is_exact() -> None:
    assert build_prompt("  What is 1 + 1?  ") == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 1 + 1?"},
    ]
    assert SYSTEM_PROMPT == (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def test_empty_prompt_fields_are_rejected() -> None:
    with pytest.raises(PromptTemplateError):
        build_prompt("   ")
    with pytest.raises(PromptTemplateError):
        build_sft_messages("question", "")


def test_generation_render_adds_one_generation_marker() -> None:
    tokenizer = FakeTokenizer()
    rendered = render_generation_prompt(tokenizer, "Question?", sample_id="sample-1")

    assert rendered.mode == "generation"
    assert rendered.sample_id == "sample-1"
    assert rendered.rendered_text.endswith("<assistant>")
    assert rendered.input_ids == (10, 20, 30)
    assert [call[2] for call in tokenizer.calls] == [True, True]


def test_sft_render_does_not_add_an_extra_generation_marker() -> None:
    tokenizer = FakeTokenizer()
    rendered = render_sft_example(tokenizer, "Question?", "\\boxed{2}")

    assert rendered.mode == "sft"
    assert rendered.messages[-1]["role"] == "assistant"
    assert not rendered.rendered_text.endswith("<assistant>")
    assert [call[2] for call in tokenizer.calls] == [False, False]


def test_rendered_prompt_debug_is_saved_without_overwrite(tmp_path) -> None:
    tokenizer = FakeTokenizer()
    rendered = render_generation_prompt(tokenizer, "Question?", sample_id="sample-1")
    path = tmp_path / "rendered_prompts.jsonl"

    save_rendered_prompt_debug([rendered], path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["rendered_text"] == rendered.rendered_text
    assert saved["input_ids"] == [10, 20, 30]

    with pytest.raises(FileExistsError):
        save_rendered_prompt_debug([rendered], path)
