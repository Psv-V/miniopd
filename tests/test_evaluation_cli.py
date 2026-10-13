import pytest
from miniopd.evaluation.cli import _load_yaml, main


def test_evaluation_cli_help(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "generate and score" in capsys.readouterr().out


def test_load_formal_evaluation_config(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        "backend: vllm\ntemperature: 0\ndo_sample: false\n"
        "max_new_tokens: 512\nbatch_size: 8\nseed: 42\n",
        encoding="utf-8",
    )
    assert _load_yaml(path) == {
        "backend": "vllm",
        "temperature": 0,
        "do_sample": False,
        "max_new_tokens": 512,
        "batch_size": 8,
        "seed": 42,
    }


def test_yaml_loader_leaves_command_specific_validation_to_caller(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text("temperature: 0\ntypo: true\n", encoding="utf-8")
    assert _load_yaml(path)["typo"] is True
