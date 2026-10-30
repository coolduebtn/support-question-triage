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
    if not (cfg.system_model and cfg.judge_model and cfg.embed_model):
        import openai

        ids = sorted(m.id for m in openai.OpenAI().models.list())
        print("Models visible to this key (chat and embedding):")
        print("\n".join(i for i in ids if i.startswith(("gpt", "o1", "o3", "o4", "text-embedding"))))
        print("Set TRIAGE_SYSTEM_MODEL, TRIAGE_JUDGE_MODEL, TRIAGE_EMBED_MODEL and TRIAGE_ANNOTATOR_MODELS in .env", file=sys.stderr)
        return 2
    wanted = {cfg.system_model, cfg.judge_model, cfg.embed_model, *cfg.annotator_models} - {""}
    import openai

    available = {m.id for m in openai.OpenAI().models.list()}
    for m in sorted(wanted):
        print(f"model {m}: {'available' if m in available else 'NOT AVAILABLE'}")
    from triage.llm import PriceMissing

    try:
        llm = LLMClient(cfg.system_model, cfg.data_dir / "cache" / "smoke", load_price(cfg.system_model))
        emb = LLMClient(cfg.embed_model, cfg.data_dir / "cache" / "smoke", load_price(cfg.embed_model))
    except PriceMissing as e:
        print(str(e), file=sys.stderr)
        return 2
    print("chat:", llm.complete([{"role": "user", "content": "Say OK."}]).text[:40])

    class Ping(BaseModel):
        ok: bool

    print("structured:", llm.complete([{"role": "user", "content": "Return ok=true."}], response_format=Ping).parsed)
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


def _collect(args: argparse.Namespace) -> int:
    import json

    from triage.config import load_settings
    from triage.sources.confluence import make_session
    from triage.sources.mailing_list import fetch_threads

    cfg = load_settings()
    if not cfg.contact:
        print("set TRIAGE_CONTACT (used in the crawler User-Agent)", file=sys.stderr)
        return 2
    threads = fetch_threads("user", "ofbiz.apache.org", window=args.window, session=make_session(cfg.contact))
    out = cfg.data_dir / "snapshot" / "threads.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t.model_dump(), sort_keys=True) + "\n" for t in threads))
    print(f"threads={len(threads)} replied={sum(t.n_replies > 0 for t in threads)} -> {out}")
    return 0


register("collect", "fetch and scrub list threads (gitignored raw snapshot)", _collect,
         lambda p: p.add_argument("--window", default="3y"))
register("ingest", "fetch the wiki snapshot (and later build indexes)", _ingest,
         lambda p: (p.add_argument("--fetch-only", action="store_true"), p.add_argument("--index", action="store_true")))
def _label(args: argparse.Namespace) -> int:
    import json
    import os
    from collections import Counter

    from triage.config import load_settings
    from triage.labelling import review as rv
    from triage.labelling.select import assign_split, select_candidates
    from triage.schemas import Question, Thread

    cfg = load_settings()
    gold = cfg.data_dir / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    qfile, splitfile = gold / "questions.jsonl", gold / "split.json"
    blind, records_file = gold / "owner_blind.md", gold / "records.jsonl"
    if args.select:
        if splitfile.exists() and not args.force:
            print(f"{splitfile} already exists; selecting again would redraw the frozen split. Use --force only before labelling.",
                  file=sys.stderr)
            return 2
        if blind.exists() and rv.read_owner_labels(blind):
            print("the blind sheet already holds owner labels; refusing to overwrite them", file=sys.stderr)
            return 2
        threads = [Thread(**json.loads(line)) for line in (cfg.data_dir / "snapshot" / "threads.jsonl").read_text().splitlines()]
        qs = select_candidates(threads, args.n, cfg.seed)
        qfile.write_text("".join(q.model_dump_json() + "\n" for q in qs))
        assign_split(qs, args.n_dev, cfg.seed, path=splitfile)
        rv.build_owner_blind_sheet(qs, args.blind, cfg.seed, blind)
        print(f"questions={len(qs)} dev={args.n_dev} -> {qfile}\nblind sheet ({args.blind} items): {blind}")
        return 0
    questions = [Question.model_validate_json(line) for line in qfile.read_text().splitlines()]
    if args.draft:
        if not os.environ.get("OPENAI_API_KEY"):
            print("set OPENAI_API_KEY (cohort key) to draft labels", file=sys.stderr)
            return 2
        from triage.ingest.chunking import page_text
        from triage.ingest.index import load_chunks
        from triage.labelling.annotate import WikiTools
        from triage.llm import LLMClient, load_price
        from triage.retrieval import Bm25Retriever, VectorRetriever
        from triage.sources.confluence import read_snapshot

        models = cfg.annotator_models
        if len(models) < 2 or not cfg.judge_model:
            print("set TRIAGE_ANNOTATOR_MODELS (two models, comma separated) and TRIAGE_JUDGE_MODEL", file=sys.stderr)
            return 2
        idx = cfg.data_dir / "index"
        emb = LLMClient(cfg.embed_model, cfg.data_dir / "cache" / "embed", load_price(cfg.embed_model))
        chunks = load_chunks(idx)
        tools = WikiTools(chunks, VectorRetriever(idx, emb.embed), Bm25Retriever(chunks))
        make = lambda m: LLMClient(m, cfg.data_dir / "cache" / "label", load_price(m))  # noqa: E731
        recs = rv.draft_labels(questions, tools, [make(m) for m in models[:2]], make(cfg.judge_model),
                               make(cfg.system_model),
                               {p.id: page_text(p) for p in read_snapshot(cfg.data_dir / "snapshot" / "wiki.jsonl")},
                               splitfile, out_path=records_file)
        print(f"drafted={len(recs)} agreed={sum(r.agreed for r in recs)} classes={Counter(r.final.label.value for r in recs)}")
        return 0
    from triage.labelling.models import LabelRecord

    records = [LabelRecord.model_validate_json(line) for line in records_file.read_text().splitlines()]
    blind_labels = rv.read_owner_labels(blind)
    if len(blind_labels) < 12:
        print(f"fill the blind sheet first ({len(blind_labels)} of at least 12 labelled): {blind}", file=sys.stderr)
        return 2
    review_file = gold / "review_sheet.md"
    if args.review:
        queue = rv.build_review_queue(records, set(blind_labels), seed=cfg.seed)
        rv.render_review_sheet(queue, records, review_file)
        print(f"review items={len(queue)} -> {review_file}")
        return 0
    if args.apply:
        review_labels = rv.read_owner_labels(review_file) if review_file.exists() else {}
        items = rv.apply_review(records, blind_labels, review_labels)
        (gold / "gold.jsonl").write_text("".join(g.model_dump_json() + "\n" for g in items))
        rv.write_gold_public(items, gold / "gold_public.jsonl")
        llm_before = {r.qid: r.final.label.value for r in records}
        ag = rv.owner_agreement({k: v.value for k, v in blind_labels.items()}, llm_before)
        print(f"gold={len(items)} classes={Counter(g.label.value for g in items)} sources={Counter(g.source for g in items)}")
        print(f"owner-vs-LLM (blind): {ag}")
        return 0
    return 2


def _label_args(p: argparse.ArgumentParser) -> None:
    for flag in ("--select", "--draft", "--review", "--apply"):
        p.add_argument(flag, action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n", type=int, default=75)
    p.add_argument("--n-dev", type=int, default=30)
    p.add_argument("--blind", type=int, default=20)


register("label", "select questions, draft labels, build review sheets, apply review", _label, _label_args)
def _need_key() -> bool:
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("set OPENAI_API_KEY (cohort key) for this command", file=sys.stderr)
        return False
    return True


def _build(args, cfg):
    """Shared by eval and ask: embed client, retriever, policy, config."""
    from triage.llm import LLMClient, load_price
    from triage.policies.llm_decision import LLMDecision, prompt_text
    from triage.policies.threshold import ThresholdRAG
    from triage.policies.vanilla import ANSWER_SYSTEM, VanillaRAG
    from triage.wiring import build_retriever, file_sha256, run_config

    emb = LLMClient(cfg.embed_model, cfg.data_dir / "cache" / "embed", load_price(cfg.embed_model))
    retriever = build_retriever(args.retriever, cfg, emb.embed)
    sys_llm = LLMClient(cfg.system_model, cfg.data_dir / "cache" / "system", load_price(cfg.system_model))
    base = VanillaRAG(sys_llm)
    prompt = ANSWER_SYSTEM
    policy = base
    if args.policy == "B":
        policy = LLMDecision(cfg.system_model, load_price(cfg.system_model), cfg.data_dir / "cache" / "decision")
        prompt = prompt_text()
    return emb, retriever, base, policy, prompt, sys_llm, run_config, file_sha256, ThresholdRAG


def _eval(args: argparse.Namespace) -> int:
    import hashlib
    import json

    from triage.config import load_settings
    from triage.evals.ablation import append_ablation_row
    from triage.evals.judge import Judge, assert_judge_differs
    from triage.evals.metrics import macro_f1
    from triage.evals.runner import run_eval
    from triage.llm import LLMClient, load_price
    from triage.policies.threshold import tune_threshold
    from triage.schemas import GoldItem
    from triage.wiring import load_limits

    if not _need_key():
        return 2
    cfg = load_settings()
    gold_file = cfg.data_dir / "gold" / "gold.jsonl"
    if not gold_file.exists():
        print(f"no gold set at {gold_file}; finish `triage label` first", file=sys.stderr)
        return 2
    gold = [GoldItem.model_validate_json(line) for line in gold_file.read_text().splitlines()]
    emb, retriever, base, policy, prompt, sys_llm, run_config, sha, Threshold = _build(args, cfg)
    threshold = args.threshold
    if args.policy == "A+":
        if threshold is None:
            threshold = tune_threshold([g for g in gold if g.split == "dev"], retriever, macro_f1,
                                       [round(0.2 + 0.02 * i, 2) for i in range(40)], load_limits(), k=cfg.k,
                                       clean=args.clean)
            print(f"tuned threshold on dev: {threshold}")
        policy = Threshold(base, threshold)
    judge = None
    if args.judge:
        assert_judge_differs(cfg.system_model, cfg.judge_model)
        judge = Judge(LLMClient(cfg.judge_model, cfg.data_dir / "cache" / "judge", load_price(cfg.judge_model)))
    config = run_config(args.policy, args.retriever, args.clean, cfg.k, cfg.system_model,
                        hashlib.sha256(prompt.encode()).hexdigest(), sha(cfg.data_dir / "snapshot" / "wiki.jsonl"),
                        sha(cfg.data_dir / "index" / "chunks.jsonl"), threshold, cfg.seed,
                        cfg.judge_model if judge else None, embed_model=cfg.embed_model, gold_hash=sha(gold_file))
    res = run_eval(policy, retriever, gold, args.split, config, judge=judge, repeats=args.repeats,
                   results_dir=cfg.results_dir)
    append_ablation_row(res, args.step or f"{args.policy} {args.retriever}", args.decision, args.notes,
                        path=cfg.results_dir / "ablation.md")
    print(json.dumps({k: v for k, v in res.metrics.items() if k != "per_class"}, indent=1, default=str))
    print(f"run={res.run_id} errors={res.n_errors} cost/q=${res.cost_per_q:.4f}")
    return 0


def _eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", choices=["A", "A+", "B"], required=True)
    p.add_argument("--retriever", choices=["vector", "hybrid", "rerank-minilm", "rerank-bge"], default="vector")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--clean", action="store_true", help="clean emails before retrieval")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--threshold", type=float)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--step", default="")
    p.add_argument("--decision", default="baseline")
    p.add_argument("--notes", default="")


def _ask(args: argparse.Namespace) -> int:
    from triage.config import load_settings
    from triage.wiring import run_ask

    if not args.text.strip():
        print(run_ask(args.text, None, None))
        return 0
    if not _need_key():
        return 2
    cfg = load_settings()
    args.policy = "B"
    _, retriever, _, policy, *_ = _build(args, cfg)
    print(run_ask(args.text, retriever, policy, k=cfg.k))
    return 0


register("eval", "run an evaluation and append a row to results/ablation.md", _eval, _eval_args)
register("ask", "triage one question (draft only, nothing is sent)", _ask,
         lambda p: (p.add_argument("text"), p.add_argument("--retriever", default="vector",
                                                           choices=["vector", "hybrid", "rerank-minilm", "rerank-bge"])))
def _report(_: argparse.Namespace) -> int:
    from triage.config import load_settings
    from triage.report import insert_ablation

    cfg = load_settings()
    insert_ablation(cfg.data_dir.parent / "README.md", cfg.results_dir / "ablation.md")
    print("README ablation table updated from results/ablation.md")
    return 0


register("report", "copy results/ablation.md into the README", _report)
register("smoke", "check models, embeddings and structured output (key day)", _smoke)


def main(argv: list[str]) -> int:
    import os

    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    from triage.config import load_dotenv

    load_dotenv()
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
