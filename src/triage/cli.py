import argparse
import sys
from collections.abc import Callable

Handler = Callable[[argparse.Namespace], int]
AddArgs = Callable[[argparse.ArgumentParser], None]
COMMANDS: dict[str, tuple[str, Handler, AddArgs | None]] = {}


def register(name: str, help_text: str, handler: Handler, add_args: AddArgs | None = None) -> None:
    COMMANDS[name] = (help_text, handler, add_args)


def _smoke(_: argparse.Namespace) -> int:
    """Key-day check: models visible, one embedding, one chat, one structured call, one Pydantic AI call."""
    import os

    from pydantic import BaseModel

    from triage.config import load_settings
    from triage.llm import LLMClient, load_price

    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY (cohort key) to run the smoke test", file=sys.stderr)
        return 2
    cfg = load_settings()
    wanted = {cfg.system_model, cfg.judge_model, cfg.embed_model, *cfg.annotator_models} - {""}
    import openai

    available = {m.id for m in openai.OpenAI().models.list()}
    for m in sorted(wanted):
        print(f"model {m}: {'available' if m in available else 'NOT AVAILABLE'}")
    llm = LLMClient(cfg.system_model, cfg.data_dir / "cache" / "smoke", load_price(cfg.system_model))
    print("chat:", llm.complete([{"role": "user", "content": "Say OK."}]).text[:40])

    class Ping(BaseModel):
        ok: bool

    print("structured:", llm.complete([{"role": "user", "content": "Return ok=true."}], response_format=Ping).parsed)
    emb = LLMClient(cfg.embed_model, cfg.data_dir / "cache" / "smoke", load_price(cfg.embed_model))
    print("embedding dims:", len(emb.embed(["hello"])[0]))
    from pydantic_ai import Agent

    result = Agent(f"openai:{cfg.system_model}", output_type=Ping).run_sync("Return ok=true.")
    print("pydantic-ai output:", result.output, "| usage is attribute:", not callable(result.usage))
    print(f"spent so far: ${llm.spent_usd + emb.spent_usd:.4f}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    import os

    from triage.config import load_settings
    from triage.sources.confluence import fetch_space_pages, make_session, read_snapshot, write_snapshot

    cfg = load_settings()
    snap = cfg.data_dir / "snapshot" / "wiki.jsonl"
    if args.index:
        if not os.environ.get("OPENAI_API_KEY"):
            print("set OPENAI_API_KEY (cohort key) to build the embedding index", file=sys.stderr)
            return 2
        from triage.ingest.chunking import chunk_page
        from triage.ingest.index import build_index
        from triage.llm import LLMClient, load_price

        chunks = [c for p in read_snapshot(snap) for c in chunk_page(p)]
        emb = LLMClient(cfg.embed_model, cfg.data_dir / "cache" / "embed", load_price(cfg.embed_model))
        info = build_index(chunks, emb.embed, cfg.data_dir / "index")
        print(f"chunks={info.n_chunks} index_hash={info.hash} spent=${emb.spent_usd:.4f}")
        return 0
    if not cfg.contact:
        print("set TRIAGE_CONTACT (used in the crawler User-Agent)", file=sys.stderr)
        return 2
    pages = list(fetch_space_pages("OFBIZ", make_session(cfg.contact)))
    digest = write_snapshot(pages, snap)
    print(f"pages={len(pages)} sha256={digest}")
    return 0


register("ingest", "fetch the wiki snapshot (and later build indexes)", _ingest,
         lambda p: (p.add_argument("--fetch-only", action="store_true"), p.add_argument("--index", action="store_true")))
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
