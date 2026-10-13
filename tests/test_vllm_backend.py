import os
import sys
import types

import pytest
from miniopd.evaluation.backends.vllm import VLLMBackend


class FakeLLM:
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    def get_tokenizer(self):
        return object()


def test_vllm_backend_enforces_deterministic_v1_scheduling(monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))

    VLLMBackend("student", dtype="bfloat16", seed=42)

    assert FakeLLM.kwargs["seed"] == 42
    assert FakeLLM.kwargs["model"] == "student"
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_vllm_backend_rejects_nondeterministic_override(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

    with pytest.raises(RuntimeError, match="requires.*MULTIPROCESSING=0"):
        VLLMBackend("student")
