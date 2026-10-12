from miniopd import __version__
from miniopd.cli import build_parser


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_parser() -> None:
    parser = build_parser()

    assert parser.prog == "miniopd"
    assert parser.parse_args([]) is not None
