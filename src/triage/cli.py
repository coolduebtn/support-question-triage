import argparse
import sys
from collections.abc import Callable

Handler = Callable[[argparse.Namespace], int]
COMMANDS: dict[str, tuple[str, Handler]] = {}


def register(name: str, help_text: str, handler: Handler) -> None:
    COMMANDS[name] = (help_text, handler)


def _smoke(_: argparse.Namespace) -> int:
    print("smoke: not implemented until Task 5")
    return 0


register("smoke", "check models, embeddings and structured output (key day)", _smoke)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="triage")
    sub = parser.add_subparsers(dest="command")
    for name, (help_text, _) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if args.command is None:
        parser.print_help()
        return 2
    return COMMANDS[args.command][1](args)


def entry() -> None:
    sys.exit(main(sys.argv[1:]))
