"""Trainer-side integration helpers."""

from miniopd.trainers.weight_sync import modules_equal, sync_module_weights

__all__ = ["modules_equal", "sync_module_weights"]
