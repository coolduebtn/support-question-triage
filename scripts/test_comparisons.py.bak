import json, statistics
from pathlib import Path
from triage.evals.metrics import macro_f1
from triage.evals.analysis import compare_pair
from triage.schemas import RunResult
ids={"A":"20261101-172648-caa31fb7-test","A+":"20261101-172650-dbb65c21-test","B":"20261101-181034-a0ce2387-test"}
P={}
for k,v in ids.items():
    P[k]=[json.loads(l) for l in (Path("results/runs")/v/"predictions.jsonl").read_text().splitlines() if l.strip()]
reps=sorted({p["repeat"] for p in P["B"]})
f=[]
for r in reps:
    rows=[p for p in P["B"] if p["repeat"]==r]
    f.append(macro_f1([p["gold"] for p in rows],[p["pred"] for p in rows]))
print("B macro-F1 per repeat:", [round(x,3) for x in f], "mean", round(statistics.fmean(f),3), "range", round(max(f)-min(f),3))
# agreement of B predictions across repeats
by={}
for p in P["B"]: by.setdefault(p["qid"],[]).append(p["pred"])
print("B questions with identical action in all 3 repeats:", sum(len(set(v))==1 for v in by.values()), "of", len(by))
def rr(k):
    rows=[p for p in P[k] if p.get("repeat",0)==0]
    return RunResult(run_id=ids[k],config_hash="x",split="test",predictions=rows,metrics={},n_errors=0,cost_per_q=0,latency_p50=None)
for a,b in [("A","A+"),("A+","B"),("A","B")]:
    c=compare_pair(rr(a),rr(b)); print(f"{b} vs {a}: delta {c['delta']:+.3f} CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] mcnemar p={c['mcnemar_p']:.4f} mde={c['mde']:.2f} claim={c['claim']}")
