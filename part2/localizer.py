"""
part2.localizer -- train + evaluate ONE localiser configuration with the same 4-block CV as the detector.

Per block:
  1. fit the detector preprocessing (normal-pressure model) on the training rows only
  2. localiser input = shape of the pressure drop (lib/loc_features)
  3. training data:
       real       steps of training-block leaks with exactly ONE active leak and leak flow >= --min-flow
                  (simultaneous leaks are left out of training; label = cluster of the leaking pipe)
       synthetic  model signatures injected into leak-free training stretches (lib/synthetic), --copies times
  4. model: MLP (softmax over K=20 clusters, cross-entropy) or "nearest" = most similar cluster
     signature (cosine, no training -> the non-neural baseline)
  5. predict cluster probabilities on the validation block (REAL data only)
Evaluation (out-of-fold over the year):
  with detector    alarm steps of the FINAL detector (W6 ensemble, frozen rule)  -> brief's metrics
  oracle           every step of every real leak (as if the detector were perfect) -> localiser alone

usage examples
  python -m part2.localizer --data both            # real + synthetic (default)
  python -m part2.localizer --data real
  python -m part2.localizer --data syn             # synthetic only (= hypothesis test of part 4)
  python -m part2.localizer --model nearest       # non-neural: most similar cluster signature
  python -m part2.localizer --model prior         # no information: clusters ranked by size
  python -m part2.localizer --mode change --windows 6,24
Outputs (outputs/part2/localizer/): runs.csv, <name>_oof.npz, <name>_per_leak.csv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from lib.config import CLUSTERS_CSV, DATA_DIR, P2_DIR, STEPS_PER_HOUR
from lib.data import load_labels, load_leaks, load_sensors, splits
from lib.features import FeatureBuilder
from lib.loc_features import loc_inputs
from lib.metrics import alarm_rule, localizer_metrics
from lib.models import predict, train_mlp
from lib.synthetic import inject, load_pipe_library

OUT = os.path.join(P2_DIR, "localizer")
DET = os.path.join(P2_DIR, "detector")
os.makedirs(OUT, exist_ok=True)


def final_detector_alarm():
    cfg = json.load(open(os.path.join(DET, "final_detector.json")))
    p = np.load(os.path.join(DET, f"{cfg['design']}_ens_oof.npz"))["p"]
    r = cfg["alarm_rule"]
    return alarm_rule(p, r["smooth_h"], r["thr"], r["persist_h"])


def cluster_centroids(lib, K, mode_center=True):
    """Mean centred+normalised day-mean signature per cluster (detectable pipes only)."""
    X = lib["sig24"].mean(axis=1)
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    C = np.zeros((K, X.shape[1]), np.float32)
    for c in range(K):
        m = (lib["cluster"] == c) & ~lib["weak"]
        if m.any():
            v = X[m].mean(0)
            C[c] = v / np.linalg.norm(v)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default="mlp", choices=["mlp", "nearest", "prior"])
    ap.add_argument("--data", default="both", choices=["real", "syn", "both"])
    ap.add_argument("--windows", default="6,24", help="averaging windows in hours")
    ap.add_argument("--mode", default="base", choices=["base", "change"])
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--copies", type=int, default=6, help="synthetic copies of the training data per block")
    ap.add_argument("--min-flow", type=float, default=2.0, help="training steps need leak flow >= this (m3/h)")
    ap.add_argument("--class-weight", action="store_true", help="re-weight clusters to equal importance")
    ap.add_argument("--tau", type=float, default=0.05, help="softmax temperature of the nearest baseline")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    windows = tuple(int(w) for w in a.windows.split(","))
    hidden = tuple(int(h) for h in a.hidden.split(",")) if a.hidden else ()
    name = a.name or (f"{a.model}_{a.mode}" if a.model != "mlp" else
                      f"mlp_{a.data}_{a.mode}_w{a.windows.replace(',', '-')}_h{a.hidden.replace(',', '-')}")

    rng = np.random.default_rng(a.seed)
    S = load_sensors()
    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    clusters = pd.read_csv(CLUSTERS_CSV).set_index("pipe_id")["cluster"]
    K = int(clusters.max()) + 1
    size = clusters.value_counts().reindex(range(K)).fillna(0).values
    lib = load_pipe_library()

    # per-step real label: cluster of the single active leak (-1 if 0 or >= 2 leaks active)
    LF = pd.read_csv(os.path.join(DATA_DIR, "leak_flows_m3h.csv"), index_col=0, parse_dates=True)
    active = LF.values > 0
    single = active.sum(axis=1) == 1
    ycl = np.full(len(idx), -1, np.int16)
    ycl[single] = clusters.reindex(LF.columns).values[active[single].argmax(axis=1)]
    flow = y["total_leak_m3h"].values

    oof = np.full((len(idx), K), np.nan, np.float32)
    info = {"params": 0, "train_s": 0.0, "n_real": 0, "n_syn": 0}
    t0 = time.time()
    for fname, tr, va in splits(idx, "cv"):
        fb = FeatureBuilder("amr").fit(S, y, tr)
        H = fb.transform(S)
        X = loc_inputs(H, fb, windows, a.mode)
        if a.model == "prior":                              # no data at all: rank clusters by size
            P = np.tile(size / size.sum(), (len(idx), 1)).astype(np.float32)
        elif a.model == "nearest":
            C = cluster_centroids(lib, K)
            Z = loc_inputs(H, fb, (max(windows),), a.mode)[:, :C.shape[1]] @ C.T / a.tau
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
        else:
            Xs, ys = [], []
            if a.data in ("real", "both"):
                r = np.where(tr & (ycl >= 0) & (flow >= a.min_flow))[0][::a.stride]
                Xs.append(X[r]); ys.append(ycl[r]); info["n_real"] += len(r)
            if a.data in ("syn", "both"):
                for _ in range(a.copies):
                    S2, yc2, fl2, _eps = inject(S, y, tr, lib, rng)
                    X2 = loc_inputs(fb.transform(S2), fb, windows, a.mode)
                    r = np.where((yc2 >= 0) & (fl2 >= a.min_flow))[0][::a.stride]
                    Xs.append(X2[r]); ys.append(yc2[r]); info["n_syn"] += len(r)
            model, inf = train_mlp(np.concatenate(Xs), np.concatenate(ys).astype(int), "multi",
                                   hidden=hidden, dropout=a.dropout, epochs=a.epochs, seed=a.seed,
                                   n_classes=K, class_weight=a.class_weight)
            P = predict(model, X, "multi")
            info["params"] = inf["params"]
            info["train_s"] += inf["train_s"]
        oof[va] = P[va]
        print(f"{fname}: done ({time.time() - t0:.0f}s)", flush=True)

    probs = np.nan_to_num(oof)
    pipe_cluster = clusters.to_dict()
    det_alarm = final_detector_alarm()
    m_det, per_det = localizer_metrics(idx, probs, det_alarm, leaks, pipe_cluster, size)
    oracle = (y["leak_active"].values > 0).astype(int)
    m_or, per_or = localizer_metrics(idx, probs, oracle, leaks, pipe_cluster, size)
    per = per_det.merge(per_or[["leak_id", "search_effort", "top1", "mean_rank"]], on="leak_id",
                        suffixes=("", "_oracle"))
    per.to_csv(os.path.join(OUT, f"{name}_per_leak.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, f"{name}_oof.npz"), probs=oof.astype(np.float16))

    row = dict(name=name, **{k: v for k, v in vars(a).items() if k != "name"},
               n_inputs=int(X.shape[1]), params=info["params"], n_real_train=info["n_real"],
               n_syn_train=info["n_syn"], train_s=round(info["train_s"], 1),
               search_effort=round(m_det["search_effort"], 3), top1=round(m_det["top1"], 3),
               search_effort_oracle=round(m_or["search_effort"], 3), top1_oracle=round(m_or["top1"], 3),
               wall_s=round(time.time() - t0, 1))
    runs = os.path.join(OUT, "runs.csv")
    pd.DataFrame([row]).to_csv(runs, mode="a", header=not os.path.exists(runs), index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== {name} ===")
    print(per.round(3).to_string(index=False))
    print(f"\nwith final detector: search effort {m_det['search_effort']:.3f}  top-1 {m_det['top1']:.3f}")
    print(f"oracle (all leak steps): search effort {m_or['search_effort']:.3f}  top-1 {m_or['top1']:.3f}")


if __name__ == "__main__":
    main()
