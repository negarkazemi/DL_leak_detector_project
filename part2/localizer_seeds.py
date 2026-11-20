"""
part2.localizer_seeds -- best localiser designs x 5 seeds (seed changes MLP init AND the synthetic leaks).
Reports mean +- std over seeds and the 5-seed ENSEMBLE (average of the cluster probabilities).

usage:  python -m part2.localizer_seeds            (~25 min)
        python -m part2.localizer_seeds --summary
Output: outputs/part2/localizer/seeds_summary.csv
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from lib.config import CLUSTERS_CSV
from lib.data import load_labels, load_leaks
from lib.metrics import localizer_metrics
from part2.localizer import OUT, final_detector_alarm

SEEDS = [0, 1, 2, 3, 4]
CANDIDATES = {
    "L_base":      "--data both",
    "L_w1-6-24":   "--data both --windows 1,6,24",
    "L_deep":      "--data both --hidden 256,128,64",
    "L_copies12":  "--data both --copies 12",
    "L_combo":     "--data both --windows 1,6,24 --hidden 256,128,64 --copies 12",
    "L_syn":       "--data syn",                 # synthetic only: needed for the part-4 hypothesis test
}
METRICS = ["search_effort", "top1", "search_effort_oracle", "top1_oracle"]


def run_all():
    for d, args in CANDIDATES.items():
        for s in SEEDS:
            if os.path.exists(os.path.join(OUT, f"{d}_s{s}_oof.npz")):
                continue
            print(f"######## {d} seed {s}: {args}", flush=True)
            subprocess.run([sys.executable, "-W", "ignore", "-m", "part2.localizer", "--name", f"{d}_s{s}",
                            "--seed", str(s), *args.split()], check=True, stdout=subprocess.DEVNULL)


def summarise():
    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    clusters = pd.read_csv(CLUSTERS_CSV).set_index("pipe_id")["cluster"]
    K = int(clusters.max()) + 1
    size = clusters.value_counts().reindex(range(K)).fillna(0).values
    alarm = final_detector_alarm()
    oracle = (y["leak_active"].values > 0).astype(int)
    runs = pd.read_csv(os.path.join(OUT, "runs.csv")).drop_duplicates("name", keep="last").set_index("name")
    rows = []
    for d in CANDIDATES:
        names = [f"{d}_s{s}" for s in SEEDS if f"{d}_s{s}" in runs.index]
        r = runs.loc[names]
        row = {"design": d, "n_seeds": len(names), "params": int(r["params"].iloc[0]),
               "train_s_per_run": round(r["train_s"].mean(), 1)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = r[m].mean(), r[m].std()
        P = np.mean([np.nan_to_num(np.load(os.path.join(OUT, f"{n}_oof.npz"))["probs"].astype(np.float32))
                     for n in names], axis=0)
        np.savez_compressed(os.path.join(OUT, f"{d}_ens_oof.npz"), probs=P.astype(np.float16))
        m1, per = localizer_metrics(idx, P, alarm, leaks, clusters.to_dict(), size)
        m2, _ = localizer_metrics(idx, P, oracle, leaks, clusters.to_dict(), size)
        per.to_csv(os.path.join(OUT, f"{d}_ens_per_leak.csv"), index=False)
        row.update(ens_search_effort=m1["search_effort"], ens_top1=m1["top1"],
                   ens_search_effort_oracle=m2["search_effort"], ens_top1_oracle=m2["top1"])
        rows.append(row)
    S = pd.DataFrame(rows).set_index("design")
    S.to_csv(os.path.join(OUT, "seeds_summary.csv"))
    pd.set_option("display.width", 250)
    show = pd.DataFrame({m: S[f"{m}_mean"].round(3).astype(str) + " ± " + S[f"{m}_std"].round(3).astype(str)
                         for m in METRICS})
    show["params"] = S["params"]
    print("\n=== single models: mean ± std over 5 seeds ===")
    print(show.to_string())
    print("\n=== 5-seed ensemble ===")
    print(S[[c for c in S.columns if c.startswith("ens_")]].round(3).to_string())


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        run_all()
    summarise()
