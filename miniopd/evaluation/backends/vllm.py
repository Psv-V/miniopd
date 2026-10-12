"""vLLM generation adapter."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from miniopd.evaluation.schema import GenerationConfig, GenerationOutput


class VLLMBackend:
    name = "vllm"

    def __init__(
        self, model_id: str, *, dtype: str = "auto", **engine_kwargs: Any
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError("vLLM backend requires the vllm package") from error
        self.model_id = model_id
        self.engine = LLM(model=model_id, dtype=dtype, **engine_kwargs)
        self.tokenizer = self.engine.get_tokenizer()

    def generate_batch(
        self,
        messages: Sequence[Sequence[Mapping[str, str]]],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        try:
            from vllm import SamplingParams
        except ImportError as error:
            raise RuntimeError("vLLM backend requires the vllm package") from error
        prompts = [
            self.tokenizer.apply_chat_template(
                list(item), tokenize=False, add_generation_prompt=True
            )
            for item in messages
        ]
        params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        started = time.perf_counter()
        requests = self.engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        outputs: list[GenerationOutput] = []
        per_request_latency = elapsed / len(requests) if requests else 0.0
        for request in requests:
            candidate = request.outputs[0]
            finish_reason = str(candidate.finish_reason or "stop")
            outputs.append(
                GenerationOutput(
                    response=candidate.text,
                    input_tokens=len(request.prompt_token_ids),
                    output_tokens=len(candidate.token_ids),
                    finish_reason=finish_reason,
                    latency_seconds=per_request_latency,
                )
            )
        return outputs
