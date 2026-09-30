import json
from pathlib import Path
import numpy as np

PATH = Path("aroma_principled/results/mea_v1/full_wrong.jsonl")

rows = [
    json.loads(x)
    for x in PATH.read_text().splitlines()
    if x.strip()
]

succ = [r for r in rows if r["success"] == 1]
fail = [r for r in rows if r["success"] == 0]

def med(xs):
    return float(np.median(xs)) if xs else float("nan")

def pct(n, d):
    return 100.0 * n / d if d else float("nan")

print("=" * 90)
print("MEA v1 DIAGNOSTIC")
print("=" * 90)

print(f"N = {len(rows)}")
print(f"success = {len(succ)}/{len(rows)} ({pct(len(succ), len(rows)):.2f}%)")
print(f"unresolved = {len(fail)}/{len(rows)} ({pct(len(fail), len(rows)):.2f}%)")

print("\nINITIAL MARGIN")
print(f"success median   = {med([r['margin_before'] for r in succ]):.6f}")
print(f"unresolved median= {med([r['margin_before'] for r in fail]):.6f}")

print("\nFINAL MARGIN — UNRESOLVED")
fm = np.array([r["margin_after"] for r in fail], dtype=float)
for q in [0, .25, .5, .75, .9, 1]:
    print(f"q{int(q*100):02d} = {np.quantile(fm, q):.6f}")

print("\nUNRESOLVED NEAR BOUNDARY")
for t in [1e-3, 1e-2, 0.05, 0.10, 0.50, 1.00]:
    n = int(np.sum(fm <= t))
    print(f"margin <= {t:g}: {n}/{len(fail)} ({pct(n, len(fail)):.2f}%)")

print("\nMARGIN REDUCTION")
for name, group in [("all", rows), ("success", succ), ("unresolved", fail)]:
    reductions = [
        (r["margin_before"] - r["margin_after"]) / r["margin_before"]
        for r in group
        if r["margin_before"] > 0
    ]
    print(f"{name:10s} median relative reduction = {med(reductions)*100:.2f}%")

print("\nCLIPPING BY ITERATION")
for it in [1, 2, 3]:
    traces = [
        r["iteration_trace"][it-1]
        for r in rows
        if len(r["iteration_trace"]) >= it
    ]
    clipped = sum(int(x["clipped"]) for x in traces)
    print(
        f"iter {it}: {clipped}/{len(traces)} clipped "
        f"({pct(clipped, len(traces)):.2f}%)"
    )

print("\nUNRESOLVED CLIPPING PATTERN")
all3 = 0
last_clip = 0
any_unclipped = 0

for r in fail:
    tr = r["iteration_trace"]
    clips = [int(x["clipped"]) for x in tr]

    if len(clips) == 3 and all(clips):
        all3 += 1
    if clips and clips[-1]:
        last_clip += 1
    if any(c == 0 for c in clips):
        any_unclipped += 1

print(f"all 3 steps clipped = {all3}/{len(fail)} ({pct(all3, len(fail)):.2f}%)")
print(f"last step clipped   = {last_clip}/{len(fail)} ({pct(last_clip, len(fail)):.2f}%)")
print(f"had unclipped step  = {any_unclipped}/{len(fail)} ({pct(any_unclipped, len(fail)):.2f}%)")

print("\nINITIAL REQUIRED-EFFORT / TRUST-BUDGET RATIO")
for name, group in [("success", succ), ("unresolved", fail)]:
    ratios = []
    for r in group:
        tr = r["iteration_trace"]
        if tr and tr[0]["tau"] > 0:
            ratios.append(tr[0]["predicted_effort"] / tr[0]["tau"])
    print(f"{name:10s} median eps*/tau = {med(ratios):.3f}")

print("\nPATH LENGTH")
print(f"success median    = {med([r['path_length'] for r in succ]):.4f}")
print(f"unresolved median = {med([r['path_length'] for r in fail]):.4f}")

print("=" * 90)
