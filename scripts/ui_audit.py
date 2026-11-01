"""Audit of the Streamlit viewer on the real saved runs: data consistency, then every question, filter, tab and split.
Prints findings; no API calls. Run from the project root: uv run python scripts/ui_audit.py"""
import json
import re
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

from triage.evals.metrics import macro_f1
from triage.ui.data import live_problem, load_bundle, verdict

ROOT = Path.cwd()
if not (ROOT / "data" / "gold" / "gold.jsonl").exists():
    raise SystemExit("needs the full gold file data/gold/gold.jsonl (not in the repository); owner-side audit only")
issues: list[str] = []


def flag(msg: str) -> None:
    issues.append(msg)
    print("  ISSUE:", msg)


PII = {"email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "phone": re.compile(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)"),
       "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")}

print("=== 1. DATA LAYER")
for split in ("dev", "test"):
    b = load_bundle(ROOT / "results", ROOT / "data", split)
    print(f"[{split}] approaches: { {k: (b.run_ids[k], len(v)) for k, v in b.predictions.items()} }")
    core = [a for a in ("A", "A+", "B") if a in b.predictions]
    if len(core) < 3:
        flag(f"{split}: missing core approaches, have {core}")
    sets = {a: set(b.predictions[a]) for a in b.predictions}
    ref = sets.get("A", set())
    for a, q in sets.items():
        if q != ref:
            flag(f"{split}: {a} covers a different question set than A ({len(q ^ ref)} differ)")
    n_gold = sum(1 for g in b._gold.values() if g["split"] == split)
    if len(ref) != n_gold:
        flag(f"{split}: A has {len(ref)} questions but gold has {n_gold}")
    shown_text = set()
    for label, rows in b.predictions.items():
        # metrics recomputed from the rows the UI shows
        m = b.metrics[label]
        f1 = macro_f1([r["gold"] for r in rows.values()], [r["pred"] for r in rows.values()])
        if abs(f1 - m["macro_f1"]) > 1e-6:
            flag(f"{split}/{label}: UI rows give macro-F1 {f1:.4f} but metrics.json says {m['macro_f1']:.4f}")
        if m.get("n_items") != len(rows):
            flag(f"{split}/{label}: metrics n_items {m.get('n_items')} vs {len(rows)} rows")
        for q, r in rows.items():
            g = b._gold.get(q)
            if g is None:
                flag(f"{split}/{label}/{q}: no gold record"); continue
            if g["label"] != r["gold"]:
                flag(f"{split}/{label}/{q}: gold in run {r['gold']} != gold file {g['label']}")
            if g["split"] != split:
                flag(f"{split}/{label}/{q}: gold says split {g['split']}")
            txt = b.question_text(q)
            if not txt.strip():
                flag(f"{split}/{label}/{q}: empty question text in UI")
            shown_text.add((q, txt))
            if not (r.get("reason") or "").strip():
                flag(f"{split}/{label}/{q}: empty reason")
            if r["pred"] == "answer" and not (r.get("draft") or "").strip():
                flag(f"{split}/{label}/{q}: ANSWER without a draft")
            if r["pred"] == "clarify" and not (r.get("missing_detail") or "").strip():
                flag(f"{split}/{label}/{q}: CLARIFY without missing_detail")
            if r["pred"] == "answer" and label == "B" and not r.get("citations"):
                flag(f"{split}/{label}/{q}: B ANSWER without citations")
            for pid in r.get("retrieved_pages", []):
                title, url = b.page(pid)
                if not url:
                    flag(f"{split}/{label}/{q}: retrieved page {pid} has no title/url in the index")
            if verdict(r) == "error":
                flag(f"{split}/{label}/{q}: row is an error")
    for q, txt in sorted(shown_text):
        for kind, rx in PII.items():
            if rx.search(txt):
                flag(f"{split}/{q}: possible {kind} visible in question text: {rx.search(txt).group(0)[:30]!r}")
    for q in ref:
        for pid in b.expected_pages(q):
            if not b.page(pid)[1]:
                flag(f"{split}/{q}: gold evidence page {pid} unresolved")
    classes = {}
    for q in ref:
        classes[b.gold(q)] = classes.get(b.gold(q), 0) + 1
    print(f"[{split}] classes: {classes}; questions: {len(ref)}")

print("\n=== 2. UI LAYER (real app, every question)")
t0 = time.time()
for split in ("dev", "test"):
    at = AppTest.from_file(str(ROOT / "src" / "triage" / "ui" / "app.py"), default_timeout=120).run()
    if split == "test":
        at.sidebar.selectbox[0].select("test").run()
    if at.exception:
        flag(f"{split}: app raised on load: {[e.value[:100] for e in at.exception]}"); continue
    tabs = [t.label for t in at.tabs]
    print(f"[{split}] tabs: {tabs}; info boxes: {[i.value[:50] for i in at.info]}; sidebar: {[c.value[:40] for c in at.sidebar.caption][:1]}")
    if tabs != ["Methodology", "Compare on a question", "Scoreboard", "Where it fails", "Try your own"]:
        flag(f"{split}: unexpected tabs {tabs}")
    sel = next(s for s in at.selectbox if s.label.startswith("Question"))
    qids = load_bundle(ROOT / "results", ROOT / "data", split).qids()
    bad = 0
    for q in qids:
        at = at.selectbox(key=sel.key).select(q).run()
        if at.exception:
            bad += 1; flag(f"{split}: exception rendering question {q}: {at.exception[0].value[:120]}")
        cards = [x for x in at.subheader]
        if len(cards) < 3:
            flag(f"{split}: question {q} rendered only {len(cards)} approach cards")
    print(f"[{split}] rendered all {len(qids)} questions in Compare, exceptions: {bad}")
    # filters
    radio = next(r for r in at.radio if r.label.startswith("Show questions"))
    for opt in radio.options:
        at = at.radio(key=radio.key).set_value(opt).run()
        n = next(s for s in at.selectbox if s.label.startswith("Question")).label if any(s.label.startswith("Question") for s in at.selectbox) else "no match"
        print(f"[{split}] filter '{opt}': {n}")
        if at.exception:
            flag(f"{split}: exception with filter {opt}")
    radio = next(r for r in at.radio if r.label.startswith("Show questions"))
    at = at.radio(key=radio.key).set_value("any").run()
    for i in range(len(at.checkbox)):
        label = at.checkbox[i].label
        at = at.checkbox[i].check().run()
        if at.exception:
            flag(f"{split}: exception with checkbox {label}")
        n = next((x.label for x in at.selectbox if x.label.startswith("Question")), "No questions match")
        print(f"[{split}] checkbox '{label}' ->", n)
        at = at.checkbox[i].uncheck().run()
    # failures tab, every approach
    ai = next(i for i, x in enumerate(at.selectbox) if x.label == "Approach")
    opts = list(at.selectbox[ai].options)
    for o in opts:
        at = at.selectbox[ai].select(o).run()
        if at.exception:
            flag(f"{split}: exception in Where it fails for {o}")
        n = [e.label for e in at.expander if e.label.startswith("correct:")]
        print(f"[{split}] Where it fails -> {o}: {len(n)} failure rows")
    # scoreboard content
    print(f"[{split}] dataframes: {len(at.dataframe)}, warnings: {[w.value[:70] for w in at.warning]}, line charts: {len(at.get('arrow_line_chart'))}, bar charts: {len(at.get('arrow_bar_chart'))}")
print(f"UI loops took {time.time() - t0:.0f}s")

print("\n=== 3. LIVE TAB readiness")
print("live_problem:", live_problem(ROOT))

print("\n=== SUMMARY:", len(issues), "issue(s)")
for i in issues:
    print(" -", i)
