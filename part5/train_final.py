"""
part5.train_final -- train the FINAL detector and localiser on ALL 2018 data and save everything
predict.py needs into models/ (the deliverable folder).

Frozen choices (part 3):
  preprocessing  FeatureBuilder("amr"), fitted on all leak-free 2018 rows
  detector       W = 6 h lags of 33 pressure z + unexplained water + hour/day-of-week, MLP 128-64,
                 real steps + 3 synthetic copies (synthetic leak steps as extra positives), 5 seeds (ensemble)
  alarm rule     from outputs/part2/detector/final_detector.json (NOT re-tuned)
  localiser      centred/normalised 6 h + 24 h pressure pattern, MLP 128-64, real single-leak steps
                 (>= 2 m3/h) + 12 synthetic copies of the leak-free hours, 5 seeds (ensemble)

usage:  python -m part5.train_final          (~5-8 min on a laptop CPU)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from lib.config import CLUSTERS_CSV, DATA_DIR, MODELS_DIR, P2_DIR
from lib.data import load_labels, load_sensors
from lib.features import FeatureBuilder, make_window
from lib.loc_features import loc_inputs
from lib.models import train_mlp
from lib.synthetic import inject, load_pipe_library

SEEDS = [0, 1, 2, 3, 4]
DET = dict(W=6, mode="lags", hidden=(128, 64), dropout=0.2, epochs=15, stride=3, syn_copies=3)
LOC = dict(windows=(6, 24), mode="base", hidden=(128, 64), dropout=0.3, epochs=15, stride=3, copies=12, min_flow=2.0)


def save_model(model, path, in_dim, hidden, out_dim, dropout):
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "hidden": list(hidden),
                "out_dim": out_dim, "dropout": dropout}, path)


def main():
    t0 = time.time()
    S = load_sensors()
    y = load_labels()
    n = len(y)
    everything = np.ones(n, bool)
    clusters = pd.read_csv(CLUSTERS_CSV).set_index("pipe_id")["cluster"]
    K = int(clusters.max()) + 1

    # ---- preprocessing, fitted on all 2018 leak-free rows
    fb = FeatureBuilder("amr").fit(S, y, everything)
    fb.save(os.path.join(MODELS_DIR, "preprocessing.pkl"))
    H = fb.transform(S)
    print(f"preprocessing fitted on {fb.params['n_fit_rows']} leak-free rows")

    # ---- detector ensemble (real steps + synthetic leak steps as extra positives)
    Xd, names = make_window(H, DET["W"], DET["mode"])
    lab = y["leak_active"].values
    rows = np.arange(n)[::DET["stride"]]
    lib = load_pipe_library()
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        xs, ys = [Xd[rows]], [lab[rows]]
        for _ in range(DET["syn_copies"]):
            S2, yc2, _fl, _ = inject(S, y, everything, lib, rng)
            X2, _ = make_window(fb.transform(S2), DET["W"], DET["mode"])
            r = np.where(yc2 >= 0)[0][::DET["stride"]]
            xs.append(X2[r]); ys.append(np.ones(len(r), lab.dtype))
        m, inf = train_mlp(np.concatenate(xs), np.concatenate(ys), "binary", hidden=DET["hidden"],
                           dropout=DET["dropout"], epochs=DET["epochs"], seed=s)
        save_model(m, os.path.join(MODELS_DIR, f"detector_s{s}.pt"), Xd.shape[1], DET["hidden"], 1, DET["dropout"])
        print(f"detector seed {s}: {inf['params']} params, {inf['train_s']:.0f}s")

    # ---- localiser ensemble
    LF = pd.read_csv(os.path.join(DATA_DIR, "leak_flows_m3h.csv"), index_col=0, parse_dates=True)
    active = LF.values > 0
    single = active.sum(axis=1) == 1
    ycl = np.full(n, -1, np.int16)
    ycl[single] = clusters.reindex(LF.columns).values[active[single].argmax(axis=1)]
    flow = y["total_leak_m3h"].values
    Xl = loc_inputs(H, fb, LOC["windows"], LOC["mode"])
    real = np.where((ycl >= 0) & (flow >= LOC["min_flow"]))[0][::LOC["stride"]]
    for s in SEEDS:
        rng = np.random.default_rng(s)
        Xs, ys = [Xl[real]], [ycl[real]]
        for _ in range(LOC["copies"]):
            S2, yc2, fl2, _ = inject(S, y, everything, lib, rng)
            X2 = loc_inputs(fb.transform(S2), fb, LOC["windows"], LOC["mode"])
            r = np.where((yc2 >= 0) & (fl2 >= LOC["min_flow"]))[0][::LOC["stride"]]
            Xs.append(X2[r]); ys.append(yc2[r])
        Xtr, ytr = np.concatenate(Xs), np.concatenate(ys).astype(int)
        m, inf = train_mlp(Xtr, ytr, "multi", hidden=LOC["hidden"], dropout=LOC["dropout"], epochs=LOC["epochs"],
                           seed=s, n_classes=K, class_weight=False)
        save_model(m, os.path.join(MODELS_DIR, f"localizer_s{s}.pt"), Xl.shape[1], LOC["hidden"], K, LOC["dropout"])
        print(f"localiser seed {s}: {len(real)} real + {len(Xtr) - len(real)} synthetic rows, "
              f"{inf['params']} params, {inf['train_s']:.0f}s")

    det_cfg = json.load(open(os.path.join(P2_DIR, "detector", "final_detector.json")))
    cfg = {"seeds": SEEDS, "n_clusters": K, "sensors": fb.params["sensors"],
           "detector": {**DET, "hidden": list(DET["hidden"]), "n_inputs": int(Xd.shape[1])},
           "alarm_rule": det_cfg["alarm_rule"],
           "localizer": {**LOC, "hidden": list(LOC["hidden"]), "windows": list(LOC["windows"]),
                         "n_inputs": int(Xl.shape[1])},
           "trained_on": "historical_2018 (all 105,120 steps)",
           "train_minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved to {MODELS_DIR} in {cfg['train_minutes']} min")


if __name__ == "__main__":
    main()
