import argparse

from miniopd import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniopd",
        description=(
            "MiniOPD: low-cost on-policy distillation for mathematical reasoning"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main() -> None:
    build_parser().parse_args()
