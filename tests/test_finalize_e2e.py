from pathlib import Path

from miniopd.diagnostics.finalize_e2e import parse_step_metrics


def test_parse_step_metrics_uses_completed_step_lines(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text(
        "noise\n(TaskRunner) step:1 - actor/loss:0.25 - training/global_step:1\n"
        "(TaskRunner) step:2 - actor/loss:0.20 - training/global_step:2\n"
    )
    second.write_text("(TaskRunner) step:3 - actor/loss:0.15 - timing_s/step:3.5\n")
    rows = parse_step_metrics([first, second])
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[0]["actor/loss"] == 0.25
    assert rows[2]["timing_s/step"] == 3.5
