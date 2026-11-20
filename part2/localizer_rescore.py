"""
part2.localizer_rescore -- recompute the "with detector" localiser metrics of every saved run after the
final detector (and therefore its alarm steps) changed. The localiser predictions themselves do not
depend on the detector, so no retraining is needed: the saved out-of-fold probabilities are re-scored.

Updates outputs/part2/localizer/runs.csv (search_effort, top1) and every <name>_per_leak.csv,
then re-runs the seed/ensemble summary.

usage:  python -m part2.localizer_rescore
"""
import os

import numpy as np
import pandas as pd

from lib.config import CLUSTERS_CSV
from lib.data import load_labels, load_leaks
from lib.metrics import localizer_metrics
from part2.localizer import OUT, final_detector_alarm


def main():
    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    clusters = pd.read_csv(CLUSTERS_CSV).set_index("pipe_id")["cluster"]
    K = int(clusters.max()) + 1
    size = clusters.value_counts().reindex(range(K)).fillna(0).values
    alarm = final_detector_alarm()
    oracle = (y["leak_active"].values > 0).astype(int)

    runs_path = os.path.join(OUT, "runs.csv")
    runs = pd.read_csv(runs_path).drop_duplicates("name", keep="last")
    for i, r in runs.iterrows():
        f = os.path.join(OUT, f"{r['name']}_oof.npz")
        if not os.path.exists(f):
            continue
        P = np.nan_to_num(np.load(f)["probs"].astype(np.float32))
        m1, per1 = localizer_metrics(idx, P, alarm, leaks, clusters.to_dict(), size)
        _, per2 = localizer_metrics(idx, P, oracle, leaks, clusters.to_dict(), size)
        runs.loc[i, "search_effort"] = round(m1["search_effort"], 3)
        runs.loc[i, "top1"] = round(m1["top1"], 3)
        per1.merge(per2[["leak_id", "search_effort", "top1", "mean_rank"]], on="leak_id",
                   suffixes=("", "_oracle")).to_csv(os.path.join(OUT, f"{r['name']}_per_leak.csv"), index=False)
    runs.to_csv(runs_path, index=False)
    print(runs[["name", "search_effort", "top1", "search_effort_oracle", "top1_oracle"]].to_string(index=False))

    import sys
    from part2 import localizer_seeds
    sys.argv = ["x", "--summary"]
    localizer_seeds.summarise()


if __name__ == "__main__":
    main()
