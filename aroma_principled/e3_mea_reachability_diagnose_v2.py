import json
from pathlib import Path
import numpy as np

P = Path("aroma_principled/results/mea_v1/horizon20.jsonl")
rows = [json.loads(x) for x in P.read_text().splitlines() if x.strip()]

succ = [r for r in rows if r["success"] == 1]
fail = [r for r in rows if r["success"] == 0]

def med(xs):
    return float(np.median(xs)) if xs else float("nan")

def switches(r):
    cs = [x["competitor_before"] for x in r["iteration_trace"]]
    return sum(a != b for a, b in zip(cs, cs[1:]))

def eps_tau0(r):
    x = r["iteration_trace"][0]
    return x["predicted_effort"] / x["tau"]

def final_margin(r):
    return float(r["margin_after"])

print("=" * 92)
print("MEA v2 REACHABILITY DIAGNOSTIC")
print("=" * 92)

print(f"N = {len(rows)}")
print(f"success = {len(succ)}/{len(rows)} ({100*len(succ)/len(rows):.2f}%)")
print(f"unresolved = {len(fail)}/{len(rows)} ({100*len(fail)/len(rows):.2f}%)")

print("\nINITIAL DIFFICULTY")
print(f"success median margin      = {med([r['margin_before'] for r in succ]):.6f}")
print(f"unresolved median margin   = {med([r['margin_before'] for r in fail]):.6f}")
print(f"success median eps*/tau    = {med([eps_tau0(r) for r in succ]):.3f}")
print(f"unresolved median eps*/tau = {med([eps_tau0(r) for r in fail]):.3f}")

print("\nUNRESOLVED FINAL MARGIN")
fm = np.array([final_margin(r) for r in fail], dtype=float)
for q in [0, .25, .5, .75, 1]:
    print(f"q{int(q*100):02d} = {np.quantile(fm, q):.6f}")

print("\nUNRESOLVED NEAR-BOUNDARY")
for t in [1e-3, 1e-2, 0.1, 0.5, 1.0]:
    n = int(np.sum(fm <= t))
    print(f"margin <= {t:g}: {n}/{len(fail)} ({100*n/len(fail):.2f}%)")

print("\nCOMPETITOR SWITCHING")
for name, group in [("success", succ), ("unresolved", fail)]:
    sw = [switches(r) for r in group]
    any_sw = sum(x > 0 for x in sw)
    print(
        f"{name:10s}: any switch = {any_sw}/{len(group)} "
        f"({100*any_sw/len(group):.2f}%), "
        f"median switches = {med(sw):.1f}"
    )

print("\n20-STEP CLIPPING — UNRESOLVED")
all_clip = 0
last_clip = 0
for r in fail:
    clips = [int(x["clipped"]) for x in r["iteration_trace"]]
    all_clip += int(len(clips) == 20 and all(clips))
    last_clip += int(bool(clips) and clips[-1] == 1)

print(f"all 20 clipped = {all_clip}/{len(fail)} ({100*all_clip/len(fail):.2f}%)")
print(f"last step clipped = {last_clip}/{len(fail)} ({100*last_clip/len(fail):.2f}%)")

print("\nEFFORT / GRADIENT AT FINAL STEP — UNRESOLVED")
last_ratio = []
last_pg = []

for r in fail:
    x = r["iteration_trace"][-1]
    if x["tau"] > 0:
        last_ratio.append(x["predicted_effort"] / x["tau"])
    if x["predicted_effort"] > 0:
        last_pg.append(abs(x["margin_before"]) / x["predicted_effort"])

print(f"median final eps*/tau = {med(last_ratio):.3f}")
print(f"median implied ||Pg|| = {med(last_pg):.6e}")

print("\nPATH LENGTH")
print(f"success median    = {med([r['path_length'] for r in succ]):.4f}")
print(f"unresolved median = {med([r['path_length'] for r in fail]):.4f}")

print("\nUNRESOLVED SAMPLE SUMMARY")
for r in fail:
    print(
        f"{r['sample_id']}  "
        f"m0={r['margin_before']:.4f}  "
        f"m20={r['margin_after']:.4f}  "
        f"switches={switches(r)}  "
        f"path={r['path_length']:.2f}"
    )

print("=" * 92)
