"""
part2.detector -- train + evaluate ONE detector configuration with cross-validation.

For every split (default: 4 event-aligned time blocks, see lib/data.splits):
   1. fit preprocessing on the training rows only           (lib/features.FeatureBuilder)
   2. build fixed-length windows x(t)                        (lib/features.make_window)
   3. train the model on training rows                       (MLP, or a non-neural baseline)
   4. predict p(t) on the validation block
The validation predictions of all blocks together cover the whole year ("out-of-fold", OOF).
Then the alarm rule (smoothing, threshold, persistence) is tuned on the OOF p(t) and all
detector metrics of the brief are computed.

usage examples
  python -m part2.detector                                   # default: amr baseline, 12 h lags, MLP 128-64
  python -m part2.detector --baseline none                   # raw pressure instead of deviation
  python -m part2.detector --W 24 --mode multiscale
  python -m part2.detector --inputs pressure                 # pressure only (no inflow)
  python -m part2.detector --model logreg                    # non-neural baseline, same inputs
  python -m part2.detector --model score                     # no training: 24 h mean of unexplained water
  python -m part2.detector --ignore-small 2                  # training label: ignore leak steps < 2 m3/h
  python -m part2.detector --split time                      # Jan-Sep / Oct-Dec instead of CV
  python -m part2.detector --baseline amr_nolevel            # normal model without tank level (or level_noamr)
  python -m part2.detector --syn-copies 3                    # + synthetic leaks as extra positives

Outputs (outputs/part2/detector/):
  runs.csv                one row per run: config + metrics (appended)
  <name>_oof.npz          OOF p(t) (used later by the localiser / final choice)
  <name>_alarm_grid.csv   all alarm rules tried
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from lib.config import P2_DIR, STEPS_PER_HOUR
from lib.data import load_labels, load_leaks, load_sensors, splits
from lib.features import BASELINES, FeatureBuilder, make_window
from lib.metrics import alarm_rule, detector_metrics, tune_alarm
from lib.models import predict, train_mlp
from lib.synthetic import inject, load_pipe_library

OUT = os.path.join(P2_DIR, "detector")
os.makedirs(OUT, exist_ok=True)


def input_cols(H, which):
    p = [c for c in H.columns if c.startswith("p_")]
    return {"pressure": p, "water": ["water"], "all": p + ["water"]}[which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--baseline", default="amr", choices=list(BASELINES))
    ap.add_argument("--W", type=int, default=12, help="window length in hours (1..24)")
    ap.add_argument("--mode", default="lags", choices=["lags", "multiscale"])
    ap.add_argument("--inputs", default="all", choices=["all", "pressure", "water"])
    ap.add_argument("--model", default="mlp", choices=["mlp", "logreg", "score"])
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=3, help="use every n-th 5-min row for training")
    ap.add_argument("--ignore-small", type=float, default=0.0,
                    help="training only: ignore leak steps with total leak flow below this (m3/h)")
    ap.add_argument("--syn-copies", type=int, default=0,
                    help="add synthetic leaks (model signatures on leak-free training hours) as extra positives")
    ap.add_argument("--split", default="cv", choices=["cv", "time"])
    ap.add_argument("--max-fa", type=float, default=3.0, help="max false alarms per 30 days when tuning the alarm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    hidden = tuple(int(h) for h in a.hidden.split(",")) if a.hidden else ()
    name = a.name or f"{a.model}_{a.baseline}_{a.inputs}_W{a.W}{a.mode[0]}" + \
        (f"_h{a.hidden.replace(',', '-')}" if a.model == "mlp" else "") + \
        (f"_ign{a.ignore_small:g}" if a.ignore_small else "") + (f"_syn{a.syn_copies}" if a.syn_copies else "") +         ("_time" if a.split == "time" else "")

    S = load_sensors()
    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    lab = y["leak_active"].values
    weight = np.ones(len(lab), np.float32)
    if a.ignore_small > 0:
        weight[(lab == 1) & (y["total_leak_m3h"].values < a.ignore_small)] = 0.0

    lib = load_pipe_library() if a.syn_copies else None
    rng = np.random.default_rng(a.seed)
    oof = np.full(len(idx), np.nan, np.float32)
    info = {"params": 0, "train_s": 0.0, "fold_auroc": []}
    t0 = time.time()
    for fname, tr, va in splits(idx, a.split):
        fb = FeatureBuilder(a.baseline).fit(S, y, tr)
        H = fb.transform(S)
        cols = input_cols(H, a.inputs)
        if a.model == "score":                      # non-neural, no training
            s = H[cols].mean(axis=1) if a.inputs == "water" else -H[cols].mean(axis=1) if a.inputs == "pressure" \
                else H["water"] - H[[c for c in cols if c.startswith("p_")]].mean(axis=1)
            s = s.rolling(24 * STEPS_PER_HOUR, min_periods=1).mean()
            p = 1 / (1 + np.exp(-s.values))
        else:
            X, _ = make_window(H, a.W, a.mode, cols)
            trs = np.where(tr)[0][::a.stride]
            Xtr, ytr, wtr = X[trs], lab[trs], weight[trs]
            if a.syn_copies:                        # extra positives: synthetic leaks on leak-free training hours
                xs, n_syn = [Xtr], 0
                for _ in range(a.syn_copies):
                    S2, yc2, fl2, _e = inject(S, y, tr, lib, rng)
                    X2, _ = make_window(fb.transform(S2), a.W, a.mode, cols)
                    r = np.where(yc2 >= 0)[0][::a.stride]
                    xs.append(X2[r]); n_syn += len(r)
                Xtr = np.concatenate(xs)
                ytr = np.concatenate([ytr, np.ones(n_syn, ytr.dtype)])
                wtr = np.concatenate([wtr, np.ones(n_syn, np.float32)])
            if a.model == "logreg":
                m = LogisticRegression(max_iter=300, class_weight="balanced", C=0.1)
                m.fit(Xtr, ytr, sample_weight=wtr)
                p = m.predict_proba(X)[:, 1]
                info["params"] = X.shape[1] + 1
            else:
                model, inf = train_mlp(Xtr, ytr, "binary", wtr, hidden=hidden,
                                       dropout=a.dropout, epochs=a.epochs, seed=a.seed)
                p = predict(model, X)
                info["params"] = inf["params"]
                info["train_s"] += inf["train_s"]
        oof[va] = p[va]
        fa = roc_auc_score(lab[va], p[va])
        info["fold_auroc"].append(round(float(fa), 3))
        print(f"{fname}: val AUROC {fa:.3f}")

    ev = ~np.isnan(oof)
    best, grid = tune_alarm(idx, np.nan_to_num(oof), y, leaks, ev, max_fa_30d=a.max_fa)
    alarm = alarm_rule(np.nan_to_num(oof), **best)
    met = detector_metrics(idx, np.nan_to_num(oof), alarm, y, leaks, ev)

    np.savez(os.path.join(OUT, f"{name}_oof.npz"), p=oof, alarm=alarm, index=idx.values.astype("datetime64[ns]").astype("int64"))
    grid.to_csv(os.path.join(OUT, f"{name}_alarm_grid.csv"), index=False)
    row = dict(name=name, **{k: v for k, v in vars(a).items() if k != "name"},
               n_inputs=None if a.model == "score" else int(X.shape[1]),
               params=info["params"], train_s=round(info["train_s"], 1),
               fold_auroc=json.dumps(info["fold_auroc"]),
               alarm_smooth_h=best["smooth_h"], alarm_thr=best["thr"], alarm_persist_h=best["persist_h"],
               **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in met.items()},
               wall_s=round(time.time() - t0, 1))
    runs = os.path.join(OUT, "runs.csv")
    pd.DataFrame([row]).to_csv(runs, mode="a", header=not os.path.exists(runs), index=False)

    print(f"\n=== {name} ===")
    print(f"OOF AUROC {met['AUROC']:.3f}   per fold {info['fold_auroc']}   params {info['params']}  "
          f"train {info['train_s']:.0f}s")
    print(f"alarm rule: mean of p over last {best['smooth_h']} h > {best['thr']}"
          + (f" for {best['persist_h']} h" if best['persist_h'] else ""))
    print(f"detected {met['detected']}/{met['n_leaks']} leaks, median delay {met['delay_h']:.1f} h, "
          f"false alarms {met['false_alarm_30d']:.2f} per 30 days ({met['fa_days']}/{met['n_free_days']} leak-free days)")


if __name__ == "__main__":
    main()
