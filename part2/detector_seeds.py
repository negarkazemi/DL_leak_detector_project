"""
part2.detector_seeds -- are the differences between the best detector designs real or luck?

Each candidate design is trained with 5 different random seeds (same CV folds).
Reported: mean +- std over seeds of every metric, and the metrics of the 5-seed ENSEMBLE
(average of the 5 p(t) curves, alarm rule re-tuned on it).

usage:  python -m part2.detector_seeds            (~25 min)
        python -m part2.detector_seeds --summary  (only re-print / re-plot from saved runs)
Outputs (outputs/part2/detector/):
  seeds_summary.csv       mean / std per design + ensemble metrics
  <design>_s<seed>_oof.npz, <design>_ens_oof.npz
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from lib.data import load_labels, load_leaks
from lib.metrics import alarm_rule, detector_metrics, tune_alarm
from part2.detector import OUT

SEEDS = [0, 1, 2, 3, 4]
CANDIDATES = {
    "W12":            "--W 12",
    "W6":             "--W 6",
    "W6_ign2":        "--W 6 --ignore-small 2",
    "W6_pressure":    "--W 6 --inputs pressure",
    "W6_deep":        "--W 6 --hidden 256,128,64",
    "W6_syn":         "--W 6 --syn-copies 3",        # + synthetic leaks as extra positives
    "W12_syn":        "--W 12 --syn-copies 3",
}
METRICS = ["AUROC", "detected", "delay_h", "false_alarm_30d"]


def run_all():
    for d, args in CANDIDATES.items():
        for s in SEEDS:
            if os.path.exists(os.path.join(OUT, f"{d}_s{s}_oof.npz")):
                continue
            print(f"\n######## {d} seed {s}: {args}", flush=True)
            subprocess.run([sys.executable, "-W", "ignore", "-m", "part2.detector",
                            "--name", f"{d}_s{s}", "--seed", str(s), *args.split()], check=True)


def summarise():
    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    runs = pd.read_csv(os.path.join(OUT, "runs.csv")).drop_duplicates("name", keep="last").set_index("name")
    rows = []
    for d in CANDIDATES:
        names = [f"{d}_s{s}" for s in SEEDS if f"{d}_s{s}" in runs.index]
        r = runs.loc[names, METRICS + ["params", "train_s"]]
        row = {"design": d, "n_seeds": len(names), "params": int(r["params"].iloc[0]),
               "train_s_per_run": round(r["train_s"].mean(), 1)}
        for m in METRICS:
            row[f"{m}_mean"] = r[m].mean()
            row[f"{m}_std"] = r[m].std()
        # ensemble: average p over seeds
        P = np.mean([np.load(os.path.join(OUT, f"{n}_oof.npz"))["p"] for n in names], axis=0)
        best, grid = tune_alarm(idx, P, y, leaks)
        a = alarm_rule(P, **best)
        m = detector_metrics(idx, P, a, y, leaks)
        np.savez(os.path.join(OUT, f"{d}_ens_oof.npz"), p=P, alarm=a)
        grid.to_csv(os.path.join(OUT, f"{d}_ens_alarm_grid.csv"), index=False)
        for k in METRICS:
            row[f"ens_{k}"] = m[k]
        row["ens_rule"] = f"mean {best['smooth_h']:g}h > {best['thr']:g}" + \
            (f" for {best['persist_h']:g}h" if best["persist_h"] else "")
        rows.append(row)
    S = pd.DataFrame(rows).set_index("design")
    S.to_csv(os.path.join(OUT, "seeds_summary.csv"))
    pd.set_option("display.width", 250)
    show = pd.DataFrame({m: S[f"{m}_mean"].round(3).astype(str) + " ± " + S[f"{m}_std"].round(3).astype(str)
                         for m in METRICS})
    show["params"] = S["params"]
    print("\n=== single models: mean ± std over 5 seeds ===")
    print(show.to_string())
    print("\n=== 5-seed ensemble (average p), alarm rule tuned on it ===")
    print(S[[f"ens_{m}" for m in METRICS] + ["ens_rule"]].round(3).to_string())


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        run_all()
    summarise()
