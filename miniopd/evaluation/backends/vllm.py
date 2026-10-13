"""vLLM generation adapter."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from miniopd.evaluation.schema import GenerationConfig, GenerationOutput


class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        dtype: str = "auto",
        seed: int = 42,
        **engine_kwargs: Any,
    ) -> None:
        multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        if multiprocessing not in {None, "0"}:
            raise RuntimeError(
                "deterministic vLLM evaluation requires "
                "VLLM_ENABLE_V1_MULTIPROCESSING=0"
            )
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError("vLLM backend requires the vllm package") from error
        self.model_id = model_id
        self.model_revision = revision
        self.engine = LLM(
            model=model_id,
            revision=revision,
            tokenizer_revision=revision,
            dtype=dtype,
            seed=seed,
            **engine_kwargs,
        )
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
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
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
