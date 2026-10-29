import argparse
import sys
from collections.abc import Callable

Handler = Callable[[argparse.Namespace], int]
AddArgs = Callable[[argparse.ArgumentParser], None]
COMMANDS: dict[str, tuple[str, Handler, AddArgs | None]] = {}


def register(name: str, help_text: str, handler: Handler, add_args: AddArgs | None = None) -> None:
    COMMANDS[name] = (help_text, handler, add_args)


def _smoke(_: argparse.Namespace) -> int:
    print("smoke: not implemented until Task 5")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from triage.config import load_settings
    from triage.sources.confluence import fetch_space_pages, make_session, write_snapshot

    cfg = load_settings()
    if not cfg.contact:
        print("set TRIAGE_CONTACT (used in the crawler User-Agent)", file=sys.stderr)
        return 2
    pages = list(fetch_space_pages("OFBIZ", make_session(cfg.contact)))
    digest = write_snapshot(pages, cfg.data_dir / "snapshot" / "wiki.jsonl")
    print(f"pages={len(pages)} sha256={digest}")
    return 0


register("ingest", "fetch the wiki snapshot (and later build indexes)", _ingest,
         lambda p: p.add_argument("--fetch-only", action="store_true"))
register("smoke", "check models, embeddings and structured output (key day)", _smoke)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="triage")
    sub = parser.add_subparsers(dest="command")
    for name, (help_text, _, add_args) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        if add_args:
            add_args(p)
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
