import json
import unicodedata

import pytest
from triage.labelling.agreement import agreement
from triage.labelling.annotate import annotate, build_prompt
from triage.labelling.models import AnnotationOut, EvidenceOut, Snippet
from triage.labelling.review import (SplitNotFrozen, apply_review, build_owner_blind_sheet,
                                     build_review_queue, draft_labels, owner_agreement, read_owner_labels)
from triage.labelling.select import assign_split, select_candidates
from triage.labelling.verify import (check_entailment, flag_possible_answer, normalise,
                                     thread_outcome_signal, verify_evidence)
from triage.llm import FakeLLM
from triage.schemas import Action, Annotation, Evidence, Question, Thread

QUOTE = "Run the reset target from the command line to restore the default publishing configuration."
PAGES = {"p1": "Intro text.  " + QUOTE + " More text."}


def thread(tid, subject, body, n_replies=0, replies=()):
    return Thread(tid=tid, subject=subject, body=body, n_replies=n_replies, epoch=1,
                  replies_text=list(replies), permalink=f"https://l/{tid}")


def q(i, text=None):
    return Question(qid=f"q{i}", thread_id=f"t{i}", text_raw=text or f"question {i}",
                    text_clean=text or f"question {i}")


# ---- select / split -------------------------------------------------------------------------

def test_select_drops_noise_and_empty_cleaned():
    threads = [thread("a", "Order PDF not working", "The PDF is blank when I print an order."),
               thread("b", "[ANNOUNCE] Apache OFBiz 18.12.09 released", "We are pleased to announce"),
               thread("c", "help", "> only quoted text\n> more")]
    out = select_candidates(threads, n_target=10, seed=1)
    assert [x.thread_id for x in out] == ["a"]
    assert out[0].text_clean.startswith("Order PDF not working")


def test_select_never_puts_replies_in_question_text():
    t = thread("a", "Export fails", "The export is failing badly here.", 1, ["SECRET_REPLY text"])
    out = select_candidates([t], 5, 1)
    assert "SECRET_REPLY" not in out[0].text_raw + out[0].text_clean


def test_select_drops_near_duplicates():
    body = "The export fails with a null pointer error when I run the invoice export job today."
    out = select_candidates([thread("a", "Export fails", body), thread("b", "Export fails", body)], 5, 1)
    assert len(out) == 1


def test_select_stratifies_by_reply_status():
    ts = [thread(f"r{i}", f"Replied question number {i}", f"unique body words alpha{i} beta{i} gamma{i} delta{i} how can I fix this?", 2)
          for i in range(6)]
    ts += [thread(f"u{i}", f"Unreplied question number {i}", f"other body terms eps{i} zeta{i} eta{i} theta{i} why does it fail?", 0)
           for i in range(6)]
    out = select_candidates(ts, n_target=4, seed=2)
    assert len(out) == 4
    assert sum(x.thread_id.startswith("r") for x in out) == 2


def test_split_deterministic_disjoint_and_written(tmp_path):
    qs = [q(i) for i in range(10)]
    path = tmp_path / "split.json"
    a = assign_split(qs, n_dev=3, seed=5, path=path)
    b = assign_split(qs, n_dev=3, seed=5)
    assert a == b and list(a.values()).count("dev") == 3 and len(a) == 10
    assert json.loads(path.read_text()) == a


# ---- annotate -------------------------------------------------------------------------------

class Tools:
    def __init__(self, cands=(), search=()):
        self.cands, self.search = list(cands), list(search)

    def candidates(self, text):
        return self.cands

    def search_wiki(self, query):
        return self.search


def snip(i, text="reset steps", score=0.2):
    return Snippet(chunk_id=f"c{i}", page_id=f"p{i}", page_title=f"Page {i}", heading="H", text=text, score=score)


def test_prompt_contains_question_and_deduped_snippets_but_no_replies():
    s = snip(1)
    prompt = build_prompt(q(1, "How do I reset?"), [s, s, snip(2)])
    assert "How do I reset?" in prompt and prompt.count("[c1]") == 1 and "[c2]" in prompt


def test_annotate_returns_annotation_from_structured_output():
    out = AnnotationOut(label="answer", evidence=[EvidenceOut(page_id="p1", heading="H", quote=QUOTE)],
                        missing_detail=None, reference_answer="Run reset.")
    llm = FakeLLM([out])
    ann = annotate(q(1), Tools([snip(1)], [snip(2)]), llm)
    assert ann.label == Action.ANSWER and ann.annotator == "fake" and ann.evidence[0].quote == QUOTE
    assert llm.calls[0]["response_format"] is AnnotationOut


# ---- verify ---------------------------------------------------------------------------------

def ann(label=Action.ANSWER, quote=QUOTE, page="p1", missing=None):
    ev = [Evidence(page_id=page, heading="H", quote=quote)] if quote else []
    return Annotation(qid="q1", annotator="a", label=label, evidence=ev, missing_detail=missing)


def test_normalise_unicode_and_whitespace():
    assert normalise("Café  TEST\n  x") == normalise(unicodedata.normalize("NFD", "café test x"))


def test_verify_normalises_whitespace_and_unicode():
    messy = QUOTE.replace(" ", "  ", 3).upper()
    out = verify_evidence(ann(quote=messy), PAGES)
    assert "evidence_failed" not in out.flags


def test_verify_flags_missing_quote_and_keeps_label():
    out = verify_evidence(ann(quote="This sentence is definitely not present in the page text at all."), PAGES)
    assert "evidence_failed" in out.flags and out.label == Action.ANSWER


def test_verify_flags_short_quote():
    out = verify_evidence(ann(quote="Intro text."), PAGES)
    assert "evidence_failed" in out.flags


def test_verify_flags_answer_without_evidence_and_unknown_page():
    assert "evidence_failed" in verify_evidence(ann(quote=None), PAGES).flags
    assert "evidence_failed" in verify_evidence(ann(page="nope"), PAGES).flags


def test_clarify_without_missing_detail_invalid():
    out = verify_evidence(ann(label=Action.CLARIFY, quote=None, missing=None), PAGES)
    assert "invalid_clarify" in out.flags
    assert "invalid_clarify" not in verify_evidence(ann(label=Action.CLARIFY, quote=None, missing="version"), PAGES).flags


def test_entailment_failure_flags_unsupported():
    out = check_entailment(ann(), "How do I reset?", FakeLLM([_V(False)]))
    assert "unsupported" in out.flags
    ok = check_entailment(ann(), "How do I reset?", FakeLLM([_V(True)]))
    assert "unsupported" not in ok.flags


def _V(b):
    from triage.labelling.models import Entailment
    return Entailment(supports=b)


def test_entailment_skipped_for_escalate():
    llm = FakeLLM([])
    out = check_entailment(ann(label=Action.ESCALATE, quote=None), "q", llm)
    assert llm.calls == [] and "unsupported" not in out.flags


def test_escalate_with_matching_snippet_flagged_possible_answer():
    a = ann(label=Action.ESCALATE, quote=None)
    hit = flag_possible_answer(a, "q text", Tools(search=[snip(1, score=0.9)]), threshold=0.6)
    assert "possible_answer" in hit.flags
    miss = flag_possible_answer(a, "q text", Tools(search=[snip(1, score=0.2)]), threshold=0.6)
    assert "possible_answer" not in miss.flags


def test_thread_outcome_signal():
    assert thread_outcome_signal(thread("a", "s", "b", 0)) == Action.ESCALATE
    assert thread_outcome_signal(thread("a", "s", "b", 1, ["Which version are you using?"])) == Action.CLARIFY
    assert thread_outcome_signal(thread("a", "s", "b", 1, ["See https://example.org/docs and run the reset target, "
                                                           "then restart the server and check the logs."])) == Action.ANSWER


# ---- agreement ------------------------------------------------------------------------------

def test_agreement_kappa_known_example():
    a = ["answer", "answer", "clarify", "escalate"]
    b = ["answer", "clarify", "clarify", "escalate"]
    out = agreement(a, b)
    assert out["raw"] == 0.75
    assert out["kappa"] == pytest.approx(0.63636, abs=1e-4)
    assert out["ac1"] == pytest.approx(0.627907, abs=1e-4)
    assert out["confusion"]["answer"]["clarify"] == 1


def test_agreement_degenerate_single_class_has_none_kappa():
    out = agreement(["answer"] * 3, ["answer"] * 3)
    assert out["raw"] == 1.0 and out["kappa"] is None


# ---- review / gold --------------------------------------------------------------------------

def test_blind_sheet_contains_questions_and_no_model_output(tmp_path):
    qs = [q(i, f"Question text {i} about invoices") for i in range(30)]
    path = build_owner_blind_sheet(qs, n=20, seed=3, path=tmp_path / "blind.md")
    text = path.read_text()
    assert text.count("owner_label:") == 20
    assert all(w not in text.lower() for w in ["annotator", "reference_answer", "confidence", "evidence"])
    again = build_owner_blind_sheet(qs, n=20, seed=3, path=tmp_path / "blind2.md").read_text()
    assert text == again


def test_read_owner_labels_parses_filled_sheet(tmp_path):
    qs = [q(i, f"Question text {i}") for i in range(3)]
    path = build_owner_blind_sheet(qs, n=3, seed=1, path=tmp_path / "blind.md")
    text = path.read_text().replace("owner_label:", "owner_label: answer", 2)
    path.write_text(text)
    labels = read_owner_labels(path)
    assert len(labels) == 2 and set(labels.values()) == {Action.ANSWER}


def _rec(i, label, flags=(), agreed=True):
    from triage.labelling.models import LabelRecord
    a = Annotation(qid=f"q{i}", annotator="a", label=label, flags=list(flags))
    return LabelRecord(qid=f"q{i}", text=f"text {i}", split="dev" if i % 2 else "test", annotations=[a, a],
                       final=a, agreed=agreed, flags=list(flags) + ([] if agreed else ["disputed"]))


def test_review_queue_has_all_flagged_and_min_3_per_class_audit():
    recs = [_rec(i, lab) for i, lab in enumerate(["answer", "clarify", "escalate"] * 10)]
    recs += [_rec(100, Action.ANSWER, agreed=False), _rec(101, Action.CLARIFY, flags=["evidence_failed"])]
    queue = build_review_queue(recs, owner_blind_qids=set(), audit_n=12, seed=4)
    by_reason = {}
    for item in queue:
        by_reason.setdefault(item.reason, []).append(item.qid)
    assert "q100" in by_reason["disputed"] and "q101" in by_reason["evidence_failed"]
    audit = [r for r in recs if r.qid in set(by_reason["audit"])]
    assert len(audit) == 12
    for lab in Action:
        assert sum(r.final.label == lab for r in audit) >= 3


def test_review_queue_excludes_owner_blind_items():
    recs = [_rec(i, Action.ANSWER) for i in range(20)]
    queue = build_review_queue(recs, owner_blind_qids={"q1", "q2"}, audit_n=5, seed=1)
    assert not ({"q1", "q2"} & {i.qid for i in queue})


def test_apply_review_sources_and_owner_precedence():
    recs = [_rec(1, Action.ANSWER), _rec(2, Action.CLARIFY), _rec(3, Action.ESCALATE), _rec(4, Action.ANSWER, agreed=False)]
    gold = apply_review(recs, owner_blind={"q1": Action.ESCALATE}, owner_review={"q2": Action.ANSWER},
                        adjudicated_qids={"q4"})
    by = {g.qid: g for g in gold}
    assert (by["q1"].label, by["q1"].source) == (Action.ESCALATE, "owner_blind")
    assert (by["q2"].label, by["q2"].source) == (Action.ANSWER, "owner_review")
    assert (by["q3"].label, by["q3"].source) == (Action.ESCALATE, "agree")
    assert by["q4"].source == "adjudicated"


def test_owner_agreement_uses_pre_review_llm_labels():
    out = owner_agreement({"q1": "answer", "q2": "clarify"}, {"q1": "answer", "q2": "escalate", "q3": "answer"})
    assert out["n"] == 2 and out["agree"] == 1 and out["rate"] == 0.5
    assert 0 <= out["ci"][0] <= 0.5 <= out["ci"][1] <= 1


# ---- pipeline -------------------------------------------------------------------------------

def _out(label, quote=QUOTE):
    ev = [EvidenceOut(page_id="p1", heading="H", quote=quote)] if label == "answer" else []
    return AnnotationOut(label=label, evidence=ev, missing_detail="version" if label == "clarify" else None,
                         reference_answer="r" if label == "answer" else None)


def test_pipeline_requires_frozen_split(tmp_path):
    with pytest.raises(SplitNotFrozen):
        draft_labels([q(1)], Tools(), [FakeLLM([]), FakeLLM([])], FakeLLM([]), FakeLLM([]), PAGES,
                     split_path=tmp_path / "missing.json")


def test_pipeline_calls_adjudicator_only_on_disagreement(tmp_path):
    qs = [q(1), q(2)]
    split = assign_split(qs, n_dev=1, seed=1, path=tmp_path / "split.json")
    a = FakeLLM([_out("answer"), _out("clarify")])
    b = FakeLLM([_out("answer"), _out("escalate")])
    adj = FakeLLM([_out("escalate")])
    checker = FakeLLM([_V(True)])
    recs = draft_labels(qs, Tools(), [a, b], adj, checker, PAGES, split_path=tmp_path / "split.json")
    assert len(adj.calls) == 1
    r1, r2 = recs
    assert r1.agreed and r1.final.label == Action.ANSWER and r1.split == split["q1"]
    assert not r2.agreed and r2.final.label == Action.ESCALATE and "disputed" in r2.flags


def test_select_keeps_only_support_style_questions():
    from triage.labelling.select import looks_like_question
    assert looks_like_question("Order PDF not working\n\nThe PDF is blank when I print an order.")
    assert looks_like_question("how to upgrade ofbiz\n\nI need steps")
    assert looks_like_question("Cost calculations\n\nHow can Ofbiz use custom cost calculation?")
    assert not looks_like_question("Looking for a french senior Ofbiz profile\n\nI am seeking a developer for a contract")
    assert not looks_like_question("Proposal: remove the eBay integration\n\nI'd like to propose removing this")
    for junk in ["Documentation Highlight: Practical Walkthrough of MRP\n\nHow it works", "[ANNOUNCEMENT] European OFBiz Days\n\nHow to join?",
                 "Participate in the ASF 25th Anniversary Campaign\n\nHow can you help?", "attempting to join mailing lists\n\nhelp",
                 "OFBIZ: new contributor request\n\nhow do I start", "Step-by-Step Guide: Calendar Function\n\nhow to",
                 "Try Apache OFBiz for Manufacturing demo\n\nhow to try", "A few small landing-page improvements\n\nwhy not"]:
        assert not looks_like_question(junk), junk
    assert not looks_like_question("Community Over Code NA 2024 Travel Assistance Applications now open!\n\nHello")


def test_wiki_tools_search_scores_by_query_term_coverage_and_candidates_union():
    from triage.labelling.annotate import WikiTools
    from triage.schemas import Chunk, Retrieved

    chunks = [Chunk(id="c1", page_id="p1", page_title="Reset", heading_path=["Reset", "Steps"],
                    text="Reset > Steps\n\nrun the reset target to restore publishing config", url="u"),
              Chunk(id="c2", page_id="p2", page_title="Pay", heading_path=["Pay"], text="Pay\n\npayment gateway setup", url="u")]

    class R:
        def __init__(self, cid):
            self.cid = cid

        def retrieve(self, query, k):
            c = next(c for c in chunks if c.id == self.cid)
            return [Retrieved(chunk=c, score=0.5, score_kind="cosine")]

    tools = WikiTools(chunks, vector=R("c1"), bm25=R("c1"))
    hits = tools.search_wiki("how do I reset the publishing config")
    assert hits[0].chunk_id == "c1" and hits[0].score >= 0.6
    assert all(h.chunk_id != "c2" for h in hits)
    assert [s.chunk_id for s in tools.candidates("anything")] == ["c1"]


def test_review_sheet_shows_llm_label_and_evidence_and_is_readable(tmp_path):
    from triage.labelling.review import render_review_sheet
    ev = Evidence(page_id="p1", heading="H", quote=QUOTE)
    a = Annotation(qid="q1", annotator="m1", label=Action.ANSWER, evidence=[ev], reference_answer="Run reset.")
    from triage.labelling.models import LabelRecord
    rec = LabelRecord(qid="q1", text="How do I reset?", split="dev", annotations=[a, a], final=a, agreed=True)
    from triage.schemas import ReviewItem
    path = render_review_sheet([ReviewItem(qid="q1", annotations=[a, a], reason="audit")], [rec], tmp_path / "r.md")
    text = path.read_text()
    assert "How do I reset?" in text and "answer" in text and QUOTE in text and "audit" in text
    path.write_text(text.replace("owner_label:", "owner_label: clarify"))
    assert read_owner_labels(path) == {"q1": Action.CLARIFY}


def test_clarify_without_any_evidence_is_flagged():
    out = verify_evidence(ann(label=Action.CLARIFY, quote=None, missing="version"), PAGES)
    assert "evidence_failed" in out.flags


def test_pipeline_survives_a_failing_annotation_and_flags_it(tmp_path):
    qs = [q(1), q(2)]
    assign_split(qs, n_dev=1, seed=1, path=tmp_path / "split.json")
    a = FakeLLM([RuntimeError("refusal"), _out("answer")])
    b = FakeLLM([_out("answer")])
    recs = draft_labels(qs, Tools(), [a, b], FakeLLM([]), FakeLLM([_V(True)]), PAGES, split_path=tmp_path / "split.json")
    assert len(recs) == 2
    assert "annotation_failed" in recs[0].flags and not recs[0].agreed
    assert recs[1].agreed


def test_pipeline_resumes_from_existing_output(tmp_path):
    qs = [q(1), q(2)]
    assign_split(qs, n_dev=1, seed=1, path=tmp_path / "split.json")
    out = tmp_path / "records.jsonl"
    first = draft_labels(qs[:1], Tools(), [FakeLLM([_out("answer")]), FakeLLM([_out("answer")])], FakeLLM([]),
                         FakeLLM([_V(True)]), PAGES, split_path=tmp_path / "split.json", out_path=out)
    a, b = FakeLLM([_out("answer")]), FakeLLM([_out("answer")])
    both = draft_labels(qs, Tools(), [a, b], FakeLLM([]), FakeLLM([_V(True)]), PAGES,
                        split_path=tmp_path / "split.json", out_path=out)
    assert [r.qid for r in both] == ["q1", "q2"] and len(a.calls) == 1 and first[0].qid == "q1"
    assert len(out.read_text().splitlines()) == 2


def test_gold_public_export_has_no_raw_text(tmp_path):
    from triage.labelling.review import write_gold_public
    from triage.schemas import GoldItem
    g = GoldItem(qid="q1", text="clean text", text_raw="RAW with signature +91 123", label=Action.ESCALATE, evidence=[],
                 missing_detail=None, reference_answer=None, split="dev", source="agree")
    path = write_gold_public([g], tmp_path / "gold_public.jsonl")
    assert "RAW" not in path.read_text() and "clean text" in path.read_text()


def test_read_sheet_qids_returns_question_ids_in_order(tmp_path):
    from triage.labelling.review import read_sheet_qids
    qs = [q(i, f"Question text {i}") for i in range(4)]
    path = build_owner_blind_sheet(qs, n=3, seed=1, path=tmp_path / "blind.md")
    ids = read_sheet_qids(path)
    assert ids == sorted(ids) and len(ids) == 3 and set(ids) <= {x.qid for x in qs}


def test_extend_candidates_adds_only_new_short_questions_and_keeps_existing_split(tmp_path):
    from triage.labelling.select import extend_candidates, extend_split
    ts = [thread(f"s{i}", f"Export {i} fails", f"How do I fix it {i}?") for i in range(8)]
    ts += [thread("long", "Long question here", "How do I fix this? " + "details " * 100)]
    existing = {"q_s0": "dev", "q_s1": "test"}
    new = extend_candidates(ts, set(existing), n=4, seed=1, max_chars=200)
    assert len(new) == 4 and all(len(x.text_clean) <= 200 for x in new) and not ({x.qid for x in new} & set(existing))
    merged = extend_split(existing, new, n_dev=1, seed=1, path=tmp_path / "split.json")
    assert merged["q_s0"] == "dev" and merged["q_s1"] == "test"
    assert sum(v == "dev" for k, v in merged.items() if k not in existing) == 1 and len(merged) == 6
