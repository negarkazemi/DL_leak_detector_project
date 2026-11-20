"""
predict.py -- leak detection + cluster-level localisation for L-Town.

    python predict.py <evaluation_folder> <output_csv>

<evaluation_folder> contains pressures_m.csv, flows_m3h.csv, levels_m.csv, demands_Lh.csv
(same file names and columns as historical_2018/, 5-min steps; no labels needed).

Output: one row per input timestamp, in time order:
    Timestamp, p_leak, alarm, c0, ..., c19
  p_leak   probability of a leak in [0, 1]       (detector: ensemble of 5 MLPs)
  alarm    0/1, frozen rule on p_leak             (frozen rule in models/config.json)
  c0..c19  cluster probabilities (sum 1, ids of clusters.csv) when alarm = 1, empty otherwise
           (localiser: ensemble of 5 MLPs)

Everything at time t uses only data at t and before. Nothing is re-fitted on the evaluation data:
all preprocessing parameters, weights and the alarm rule are loaded from models/.
Runs on CPU in well under a minute for one year of data.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lib.data import load_sensors                      # noqa: E402
from lib.features import FeatureBuilder, make_window   # noqa: E402
from lib.loc_features import loc_inputs                # noqa: E402
from lib.metrics import alarm_rule                     # noqa: E402
from lib.models import MLP, predict                    # noqa: E402

MODELS = os.path.join(ROOT, "models")


def load_models(prefix, seeds):
    out = []
    for s in seeds:
        d = torch.load(os.path.join(MODELS, f"{prefix}_s{s}.pt"), map_location="cpu")
        m = MLP(d["in_dim"], tuple(d["hidden"]), d["out_dim"], d["dropout"])
        m.load_state_dict(d["state_dict"])
        m.eval()
        out.append(m)
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python predict.py <evaluation_folder> <output_csv>")
    folder, out_csv = sys.argv[1], sys.argv[2]
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = json.load(open(os.path.join(MODELS, "config.json")))

    S = load_sensors(folder)
    S["P"] = S["P"][cfg["sensors"]]                       # same column order as in training
    fb = FeatureBuilder.load(os.path.join(MODELS, "preprocessing.pkl"))
    H = fb.transform(S)

    # ---- detector
    dc = cfg["detector"]
    Xd, _ = make_window(H, dc["W"], dc["mode"])
    assert Xd.shape[1] == dc["n_inputs"], "detector input size differs from training"
    p = np.mean([predict(m, Xd, "binary") for m in load_models("detector", cfg["seeds"])], axis=0)
    r = cfg["alarm_rule"]
    alarm = alarm_rule(p, r["smooth_h"], r["thr"], r["persist_h"])

    # ---- localiser (only needed where alarm = 1)
    K = cfg["n_clusters"]
    C = np.full((len(p), K), np.nan)
    on = alarm > 0
    if on.any():
        lc = cfg["localizer"]
        Xl = loc_inputs(H, fb, tuple(lc["windows"]), lc["mode"])
        probs = np.mean([predict(m, Xl[on], "multi") for m in load_models("localizer", cfg["seeds"])], axis=0)
        C[on] = probs / probs.sum(axis=1, keepdims=True)

    out = pd.DataFrame(C, index=H.index, columns=[f"c{k}" for k in range(K)])
    out.insert(0, "alarm", alarm.astype(int))
    out.insert(0, "p_leak", np.clip(p, 0, 1))
    out.index.name = "Timestamp"
    out.sort_index().to_csv(out_csv, float_format="%.6f")
    print(f"wrote {len(out)} rows to {out_csv} ({on.mean() * 100:.1f}% with alarm) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
