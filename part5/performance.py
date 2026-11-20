"""
part5.performance -- predicted performance on the instructor's held-out data, WITH intervals.

Basis: the out-of-fold (4-block CV) predictions of the frozen final designs
  detector  = W6 5-seed ensemble + frozen alarm rule
  localiser = L_copies12 5-seed ensemble
Point estimate = OOF value. Interval = 90 % bootstrap interval, resampling the natural units:
  leak events (detected share, delay, search effort, top-1),  leak-free days (false alarms),
  whole days (AUROC, block bootstrap so correlated 5-min steps stay together).
Also reported: the spread over the 4 CV blocks (a rough "different season" sensitivity).

usage:  python -m part5.performance
Outputs: models/performance_prediction.json (deliverable), outputs/part5/bootstrap_samples.csv
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.config import CLUSTERS_CSV, MODELS_DIR, OUT_DIR, P2_DIR
from lib.data import load_labels, load_leaks, splits
from lib.metrics import alarm_rule, detector_metrics, localizer_metrics

B = 2000
B_AUC = 300
P5 = os.path.join(OUT_DIR, "part5")
os.makedirs(P5, exist_ok=True)


def q(x, lo=5, hi=95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def clopper_pearson(k, n, conf=0.90):
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return [float(lo), float(hi)]


def main():
    rng = np.random.default_rng(0)
    y = load_labels()
    leaks = load_leaks()
    idx = y.index
    lab = y["leak_active"].values
    det = json.load(open(os.path.join(P2_DIR, "detector", "final_detector.json")))
    p = np.load(os.path.join(P2_DIR, "detector", f"{det['design']}_ens_oof.npz"))["p"]
    r = det["alarm_rule"]
    alarm = alarm_rule(p, r["smooth_h"], r["thr"], r["persist_h"])
    probs = np.load(os.path.join(P2_DIR, "localizer", "L_copies12_ens_oof.npz"))["probs"].astype(np.float32)
    probs = np.nan_to_num(probs)
    clusters = pd.read_csv(CLUSTERS_CSV).set_index("pipe_id")["cluster"]
    size = clusters.value_counts().reindex(range(int(clusters.max()) + 1)).fillna(0).values

    # ---------------- point estimates
    dm = detector_metrics(idx, p, alarm, y, leaks)
    lm, per = localizer_metrics(idx, probs, alarm, leaks, clusters.to_dict(), size)

    # per-leak table: detected / delay
    al = pd.Series(alarm, index=idx)
    d_rows = []
    for _, l in leaks.iterrows():
        a = al[l["start"]:l["end"]]
        hit = (a > 0).any()
        d_rows.append(dict(leak_id=l["leak_id"], detected=float(hit),
                           delay_h=(a.index[a.values > 0][0] - l["start"]).total_seconds() / 3600 if hit else np.nan))
    perleak = pd.DataFrame(d_rows).merge(per[["leak_id", "search_effort", "top1"]], on="leak_id")

    # leak-free qualifying days with / without alarm
    after = np.zeros(len(idx), bool)
    for _, l in leaks.iterrows():
        after |= np.asarray((idx > l["end"]) & (idx <= l["end"] + pd.Timedelta(hours=24)))
    qmask = (lab == 0) & ~after
    fa_day = pd.Series(alarm[qmask] > 0, index=idx[qmask]).groupby(idx[qmask].date).any().values

    # ---------------- bootstrap
    L = len(perleak)
    bs = {k: [] for k in ["detected_share", "delay_h", "search_effort", "top1", "false_alarm_30d"]}
    for _ in range(B):
        s = perleak.iloc[rng.integers(0, L, L)]
        bs["detected_share"].append(s["detected"].mean())
        bs["delay_h"].append(np.nanmedian(s["delay_h"]))
        bs["search_effort"].append(s["search_effort"].mean())
        bs["top1"].append(s["top1"].mean())
        bs["false_alarm_30d"].append(fa_day[rng.integers(0, len(fa_day), len(fa_day))].mean() * 30)
    # AUROC: block bootstrap over days, hourly subsample
    sub = np.arange(0, len(idx), 12)
    days = idx[sub].normalize()
    ud = np.unique(days)
    pos = {d: np.where(days == d)[0] for d in ud}
    aucs = []
    for _ in range(B_AUC):
        pick = np.concatenate([pos[d] for d in ud[rng.integers(0, len(ud), len(ud))]])
        yy, pp = lab[sub][pick], p[sub][pick]
        if 0 < yy.mean() < 1:
            aucs.append(roc_auc_score(yy, pp))
    pd.DataFrame({k: pd.Series(v) for k, v in {**bs, "AUROC": aucs}.items()}).to_csv(
        os.path.join(P5, "bootstrap_samples.csv"), index=False)

    # ---------------- spread over CV blocks
    blocks = []
    for name, _tr, va in splits(idx, "cv"):
        m = detector_metrics(idx, p, alarm, y, leaks, va)
        lm_b, _ = localizer_metrics(idx, probs, alarm, leaks, clusters.to_dict(), size, mask=va)
        blocks.append(dict(block=name, AUROC=m["AUROC"], detected=f"{m['detected']}/{m['n_leaks']}",
                           delay_h=m["delay_h"], false_alarm_30d=m["false_alarm_30d"],
                           search_effort=lm_b["search_effort"], top1=lm_b["top1"]))
    blocks = pd.DataFrame(blocks)
    blocks.to_csv(os.path.join(P5, "per_block.csv"), index=False)

    pred = {
        "basis": "out-of-fold predictions of the frozen designs (4 event-aligned time blocks of 2018); "
                 "90% bootstrap intervals (leak events / leak-free days / day blocks)",
        "AUROC": {"expected": dm["AUROC"], "interval_90": q(aucs)},
        "detected_share": {"expected": dm["detected"] / dm["n_leaks"],
                           "interval_90": clopper_pearson(dm["detected"], dm["n_leaks"]),
                           "note": f"{dm['detected']}/{dm['n_leaks']} in 2018; exact binomial (Clopper-Pearson) "
                                   "interval, because the bootstrap is degenerate when every leak is detected"},
        "detection_delay_median_h": {"expected": dm["delay_h"], "interval_90": q(bs["delay_h"])},
        "false_alarms_per_30_days": {"expected": dm["false_alarm_30d"], "interval_90": q(bs["false_alarm_30d"]),
                                     "note": f"{dm['fa_days']} of {dm['n_free_days']} leak-free days"},
        "search_effort": {"expected": lm["search_effort"], "interval_90": q(bs["search_effort"])},
        "top1_accuracy": {"expected": lm["top1"], "interval_90": q(bs["top1"])},
        "per_cv_block_range": {c: [float(blocks[c].min()), float(blocks[c].max())]
                               for c in ["AUROC", "delay_h", "false_alarm_30d", "search_effort", "top1"]},
    }
    with open(os.path.join(MODELS_DIR, "performance_prediction.json"), "w") as f:
        json.dump(pred, f, indent=2)
    pd.set_option("display.width", 200)
    print(blocks.round(3).to_string(index=False))
    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
