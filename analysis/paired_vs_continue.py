"""Paired differences against Continue from data/exp48-branches.csv.

For each stall state and arm, solve rate and best score are averaged over completed repeats. Each arm is
compared with Continue over the states where both have a result. 95% intervals are percentile intervals
from a bootstrap that resamples the 25 origin clusters (an original stall and its resampled states form
one cluster) 2,000 times. Matches Table 1 of the preprint up to bootstrap noise in the last digit.

Usage: python analysis/paired_vs_continue.py [--population recoverable|ceiling|early_origin]
"""
import argparse, collections, csv, pathlib, random, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--population", choices=["recoverable", "ceiling", "early_origin"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = list(csv.DictReader(open(pathlib.Path(__file__).resolve().parent.parent / "data" / "exp48-branches.csv")))
if args.population:
    rows = [r for r in rows if r["recoverability"] == args.population]
cell = collections.defaultdict(list)
cluster, order = {}, []
for r in rows:
    cell[(r["state"], r["arm"])].append((int(r["solved"]), float(r["best_score"])))
    cluster[r["state"]] = r["origin_cluster"]
    if r["arm"] not in order:
        order.append(r["arm"])
mean = {k: (statistics.mean(s for s, _ in v), statistics.mean(b for _, b in v)) for k, v in cell.items()}
clusters = sorted(set(cluster.values()))
rng = random.Random(args.seed)
draws = [[rng.choice(clusters) for _ in clusters] for _ in range(2000)]

def paired(arm, metric):
    per_state = {s: mean[(s, arm)][metric] - mean[(s, "Continue")][metric]
                 for (s, a) in mean if a == arm and (s, "Continue") in mean}
    by_cluster = collections.defaultdict(list)
    for s, d in per_state.items():
        by_cluster[cluster[s]].append(d)
    boot = []
    for draw in draws:
        vals = [d for c in draw for d in by_cluster.get(c, [])]
        if vals:
            boot.append(statistics.mean(vals))
    boot.sort()
    return statistics.mean(per_state.values()), boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1], len(per_state)

base = [v for (s, a), v in mean.items() if a == "Continue"]
print(f"Continue: {len(base)} states, solve {statistics.mean(s for s, _ in base):.3f}, best {statistics.mean(b for _, b in base):.3f}")
print(f"{'arm':34} {'states':>6}  {'d solve [95% CI]':>24}  {'d best [95% CI]':>24}")
for arm in sorted(order, key=lambda a: (a != "Continue", a)):
    if arm == "Continue":
        continue
    ds, ls, hs, n = paired(arm, 0)
    db, lb, hb, _ = paired(arm, 1)
    flag = lambda lo, hi: "*" if lo > 0 or hi < 0 else " "
    print(f"{arm:34} {n:>6}  {ds:+.3f} [{ls:+.2f}, {hs:+.2f}]{flag(ls, hs)}  {db:+.3f} [{lb:+.2f}, {hb:+.2f}]{flag(lb, hb)}")
print("* interval excludes zero")
