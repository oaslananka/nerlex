from __future__ import annotations

import argparse
import json
import platform
import sys

from nerlex import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nerlex",
        description="Nerlex AI Decision Compiler",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the Nerlex version")
    subparsers.add_parser("doctor", help="Print local runtime diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "nerlex_version": __version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "executable": sys.executable,
                },
                indent=2,
            )
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
