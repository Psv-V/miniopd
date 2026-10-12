"""Lazy-loaded model generation backends."""

from miniopd.evaluation.backends.transformers import TransformersBackend
from miniopd.evaluation.backends.vllm import VLLMBackend

__all__ = ["TransformersBackend", "VLLMBackend"]
