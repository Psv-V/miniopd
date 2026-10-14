import pytest
from miniopd.integration.reward_score import compute_score


def test_reward_score_uses_shared_boxed_answer_semantics() -> None:
    assert compute_score("openai/gsm8k", r"Work. \boxed{76}", "#### 76") == 1.0
    assert compute_score("openai/gsm8k", r"Work. \boxed{75}", "#### 76") == 0.0


def test_reward_score_rejects_unreviewed_data_sources() -> None:
    with pytest.raises(ValueError, match="unsupported MiniOPD reward data source"):
        compute_score("unknown", r"\boxed{1}", "1")
