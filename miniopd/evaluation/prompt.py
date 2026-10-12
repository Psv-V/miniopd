"""Shared chat prompt contract for every MiniOPD stage."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


class PromptTemplateError(ValueError):
    """Raised when a prompt or rendered chat template violates the contract."""


class ChatTemplateTokenizer(Protocol):
    """The subset of the Hugging Face tokenizer API used by MiniOPD."""

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class RenderedPrompt:
    """A JSON-serializable record of the prompt actually seen by a model."""

    mode: str
    messages: tuple[dict[str, str], ...]
    rendered_text: str
    input_ids: tuple[int, ...]
    sample_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "mode": self.mode,
            "messages": [dict(message) for message in self.messages],
            "rendered_text": self.rendered_text,
            "input_ids": list(self.input_ids),
        }


def _clean_nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptTemplateError(f"{field} must be a non-empty string")
    return value.strip()


def build_prompt(question: str) -> list[dict[str, str]]:
    """Build the exact system/user messages used for generation and evaluation."""
    question = _clean_nonempty(question, "question")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def build_sft_messages(question: str, assistant_response: str) -> list[dict[str, str]]:
    """Build the same prompt followed by one verified teacher response."""
    response = _clean_nonempty(assistant_response, "assistant_response")
    return [*build_prompt(question), {"role": "assistant", "content": response}]


def _coerce_input_ids(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(token_id, int) for token_id in value
    ):
        raise PromptTemplateError(
            "tokenizer.apply_chat_template(tokenize=True) must return token IDs"
        )
    return tuple(value)


def _render(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
    *,
    mode: str,
    add_generation_prompt: bool,
    sample_id: str | None,
) -> RenderedPrompt:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str) or not text:
        raise PromptTemplateError(
            "tokenizer.apply_chat_template(tokenize=False) must return text"
        )
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return RenderedPrompt(
        sample_id=sample_id,
        mode=mode,
        messages=tuple(dict(message) for message in messages),
        rendered_text=text,
        input_ids=_coerce_input_ids(input_ids),
    )


def render_generation_prompt(
    tokenizer: ChatTemplateTokenizer,
    question: str,
    *,
    sample_id: str | None = None,
) -> RenderedPrompt:
    """Render a prompt with the tokenizer's assistant-generation marker."""
    return _render(
        tokenizer,
        build_prompt(question),
        mode="generation",
        add_generation_prompt=True,
        sample_id=sample_id,
    )


def render_sft_example(
    tokenizer: ChatTemplateTokenizer,
    question: str,
    assistant_response: str,
    *,
    sample_id: str | None = None,
) -> RenderedPrompt:
    """Render a complete SFT conversation without another generation marker."""
    return _render(
        tokenizer,
        build_sft_messages(question, assistant_response),
        mode="sft",
        add_generation_prompt=False,
        sample_id=sample_id,
    )


def save_rendered_prompt_debug(records: Sequence[RenderedPrompt], path: Path) -> None:
    """Atomically save sampled rendered prompts for later template debugging."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite rendered prompts: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
