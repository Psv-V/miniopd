"""Hugging Face Transformers generation adapter."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from miniopd.evaluation.schema import GenerationConfig, GenerationOutput


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        dtype: str = "auto",
        **model_kwargs: Any,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Transformers backend requires torch and transformers"
            ) from error
        self._torch = torch
        self.model_id = model_id
        self.model_revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype, **model_kwargs
        )
        self.model.eval()

    def generate_batch(
        self,
        messages: Sequence[Sequence[Mapping[str, str]]],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        prompts = [
            self.tokenizer.apply_chat_template(
                list(item), tokenize=False, add_generation_prompt=True
            )
            for item in messages
        ]
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else None,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        prompt_width = encoded["input_ids"].shape[1]
        outputs: list[GenerationOutput] = []
        per_request_latency = elapsed / len(generated) if len(generated) else 0.0
        for row, input_length in zip(generated, input_lengths, strict=True):
            token_ids = row[prompt_width:].tolist()
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id != self.tokenizer.pad_token_id
            ]
            output_tokens = len(token_ids)
            outputs.append(
                GenerationOutput(
                    response=self.tokenizer.decode(token_ids, skip_special_tokens=True),
                    input_tokens=int(input_length),
                    output_tokens=output_tokens,
                    finish_reason=(
                        "length" if output_tokens >= config.max_new_tokens else "stop"
                    ),
                    latency_seconds=per_request_latency,
                )
            )
        return outputs
