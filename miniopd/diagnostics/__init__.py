"""Training-only observability that never contributes to the OPD loss."""

from miniopd.diagnostics.opd import (
    DiagnosticsConfig,
    DiagnosticsRecorder,
    compute_periodic_diagnostics,
    position_segment_masks,
    summarize_step_metrics,
)

__all__ = [
    "DiagnosticsConfig",
    "DiagnosticsRecorder",
    "compute_periodic_diagnostics",
    "position_segment_masks",
    "summarize_step_metrics",
]
