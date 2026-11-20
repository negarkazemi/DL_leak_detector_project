"""
part2.detector_choose -- freeze the final detector design + alarm rule (part 3), with evidence.

Input: a design name from part2.detector_seeds (its 5-seed ensemble OOF predictions).
  1. trade-off plot: for every false-alarm budget, the best achievable #detected and delay
     (from the alarm-rule grid) -> shows why the chosen budget is reasonable
  2. per-leak table: detected? delay? for the frozen rule (which leaks are hard)
  3. writes outputs/part2/detector/final_detector.json  (design args + frozen alarm rule)

usage:  python -m part2.detector_choose W6 [--max-fa 3]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.data import load_labels, load_leaks
from lib.metrics import alarm_rule, detector_metrics, tune_alarm
from part2.detector import OUT
from part2.detector_seeds import CANDIDATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("--max-fa", type=float, default=3.0)
    a = ap.parse_args()

    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    P = np.load(os.path.join(OUT, f"{a.design}_ens_oof.npz"))["p"]
    best, grid = tune_alarm(idx, P, y, leaks, max_fa_30d=a.max_fa)

    # ---- 1. trade-off: best achievable detection / delay for each false-alarm budget
    budgets = np.arange(0, 8.01, 0.5)
    tr = []
    for b in budgets:
        ok = grid[grid["false_alarm_30d"] <= b + 1e-9]
        if ok.empty:
            tr.append((b, np.nan, np.nan))
            continue
        top = ok.sort_values(["detected", "delay_h"], ascending=[False, True]).iloc[0]
        tr.append((b, top["detected"], top["delay_h"]))
    tr = pd.DataFrame(tr, columns=["fa_budget_30d", "best_detected", "delay_h"])
    tr.to_csv(os.path.join(OUT, f"{a.design}_tradeoff.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].step(tr["fa_budget_30d"], tr["best_detected"], where="post")
    ax[0].set(xlabel="allowed false alarms per 30 days", ylabel="leaks detected (of 20)",
              title="Detection vs false-alarm budget")
    ax[1].step(tr["fa_budget_30d"], tr["delay_h"], where="post", color="C1")
    ax[1].set(xlabel="allowed false alarms per 30 days", ylabel="median delay (h)", title="Delay vs budget")
    for x in ax:
        x.axvline(a.max_fa, color="r", ls="--", lw=1)
        x.grid(alpha=0.3)
    fig.suptitle(f"Alarm-rule trade-off, detector '{a.design}' (5-seed ensemble, out-of-fold)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{a.design}_tradeoff.png"), dpi=140)

    # ---- 2. per-leak result with the frozen rule
    alarm = alarm_rule(P, **best)
    met = detector_metrics(idx, P, alarm, y, leaks)
    al = pd.Series(alarm, index=idx)
    pr = pd.Series(P, index=idx)
    rows = []
    for _, l in leaks.iterrows():
        seg = al[l["start"]:l["end"]]
        first = seg.index[seg.values > 0][0] if (seg > 0).any() else None
        rows.append(dict(leak_id=l["leak_id"], pipe=l["pipe_id"], type=l["type"], D_cm=round(l["diameter_m"] * 100, 2),
                         duration_d=round((l["end"] - l["start"]).total_seconds() / 86400, 1),
                         detected=first is not None,
                         delay_h=round((first - l["start"]).total_seconds() / 3600, 1) if first is not None else None,
                         share_of_leak_alarmed=round(float(seg.mean()), 2),
                         mean_p=round(float(pr[l["start"]:l["end"]].mean()), 2)))
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, f"{a.design}_per_leak.csv"), index=False)

    final = {"design": a.design, "args": CANDIDATES[a.design], "n_seeds_ensemble": 5,
             "alarm_rule": {k: float(v) for k, v in best.items()}, "max_fa_30d_budget": a.max_fa,
             "oof_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                             for k, v in met.items()}}
    with open(os.path.join(OUT, "final_detector.json"), "w") as f:
        json.dump(final, f, indent=2)

    pd.set_option("display.width", 200)
    print("=== trade-off ===")
    print(tr.to_string(index=False))
    print("\n=== per leak (frozen rule) ===")
    print(per.to_string(index=False))
    print("\n=== final detector ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
