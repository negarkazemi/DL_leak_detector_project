"""
lib/metrics.py -- the evaluation metrics of the project brief, plus the alarm rule.

Detector:
  AUROC            quality of p(t) against leak_active, per 5-min step
  detected         # leak events with >= 1 alarm between their start and end
  delay_h          median hours from leak start to first alarm (detected leaks only)
  false_alarm_30d  among steps without a leak and not within 24 h after a repair:
                   share of days that contain an alarm, x 30
Localiser:
  search_effort    # pipes in the clusters searched up to the correct one (all of them included) / 905;
                   averaged over leak events; an undetected leak counts as 1
  top1_accuracy    share of alarm steps whose most likely cluster contains the leak;
                   averaged over leak events; an undetected leak counts as 0
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import STEPS_PER_HOUR


# ------------------------------------------------------------------ localiser metrics
def localizer_metrics(index, probs, alarm, leaks, pipe_cluster, cluster_size, n_pipes=905, mask=None):
    """
    Per leak event (counted if it starts inside mask), over the ALARM steps between its start and end:
      search_effort = mean over those steps of (# pipes in all clusters ranked up to and including the
                      leak's cluster) / n_pipes;  undetected leak (no alarm step) -> 1
      top1          = share of those steps whose top-ranked cluster is the leak's cluster; undetected -> 0
    Both are averaged with equal weight over leak events.  Also returns the same two numbers per leak.
    probs: [n x K] cluster probabilities (rows without alarm are ignored).
    """
    index = pd.DatetimeIndex(index)
    mask = np.ones(len(index), bool) if mask is None else np.asarray(mask)
    lo, hi = index[mask].min(), index[mask].max()
    size = np.asarray(cluster_size, float)
    rows = []
    for _, l in leaks[(leaks["start"] >= lo) & (leaks["start"] <= hi)].iterrows():
        sel = np.asarray((index >= l["start"]) & (index <= l["end"])) & (np.asarray(alarm) > 0)
        c = int(pipe_cluster[l["pipe_id"]])
        if not sel.any():
            rows.append(dict(leak_id=l["leak_id"], cluster=c, n_alarm_steps=0, search_effort=1.0, top1=0.0,
                             mean_rank=np.nan))
            continue
        pr = np.asarray(probs)[sel]
        order = np.argsort(-pr, axis=1)                         # clusters, most likely first
        rank = np.argmax(order == c, axis=1)                    # 0 = correct cluster first
        searched = np.array([size[o[: r + 1]].sum() for o, r in zip(order, rank)])
        rows.append(dict(leak_id=l["leak_id"], cluster=c, n_alarm_steps=int(sel.sum()),
                         search_effort=float(searched.mean() / n_pipes), top1=float((rank == 0).mean()),
                         mean_rank=float(rank.mean() + 1)))
    per = pd.DataFrame(rows)
    return {"search_effort": float(per["search_effort"].mean()), "top1": float(per["top1"].mean())}, per


# ------------------------------------------------------------------ alarm rule
def alarm_rule(p, smooth_h=6, thr=0.8, persist_h=0):
    """
    p(t) -> alarm(t) in {0,1}, causal:
      1. p_s = mean of p over the last smooth_h hours
      2. alarm when p_s > thr, and (if persist_h > 0) it has stayed above thr for the last persist_h hours
    """
    p = pd.Series(np.asarray(p, dtype=float))
    ps = p.rolling(max(1, int(smooth_h * STEPS_PER_HOUR)), min_periods=1).mean()
    above = (ps > thr).astype(float)
    if persist_h > 0:
        above = above.rolling(int(persist_h * STEPS_PER_HOUR), min_periods=1).min()
    return above.values.astype(np.int8)


# ------------------------------------------------------------------ detector metrics
def detector_metrics(index, p, alarm, labels, leaks, mask=None):
    """All detector metrics on the rows in mask (default: all). Leaks are counted if they start inside mask."""
    index = pd.DatetimeIndex(index)
    mask = np.ones(len(index), bool) if mask is None else np.asarray(mask)
    y = labels["leak_active"].values
    out = {}
    yy, pp = y[mask], np.asarray(p)[mask]
    out["AUROC"] = roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else np.nan

    al = pd.Series(np.asarray(alarm), index=index)
    lo, hi = index[mask].min(), index[mask].max()
    det, delays = 0, []
    ev = leaks[(leaks["start"] >= lo) & (leaks["start"] <= hi)]
    for _, l in ev.iterrows():
        a = al[l["start"]:l["end"]]
        if (a > 0).any():
            det += 1
            delays.append((a.index[a.values > 0][0] - l["start"]).total_seconds() / 3600)
    out["n_leaks"] = len(ev)
    out["detected"] = det
    out["delay_h"] = float(np.median(delays)) if delays else np.nan

    # false alarms: leak-free steps, not within 24 h after any repair
    after_repair = np.zeros(len(index), bool)
    for _, l in leaks.iterrows():
        after_repair |= np.asarray((index > l["end"]) & (index <= l["end"] + pd.Timedelta(hours=24)))
    q = mask & (y == 0) & ~after_repair
    if q.any():
        d = pd.Series(np.asarray(alarm)[q] > 0, index=index[q]).groupby(index[q].date).any()
        out["fa_days"] = int(d.sum())
        out["n_free_days"] = int(len(d))
        out["false_alarm_30d"] = float(d.mean() * 30)
    else:
        out["fa_days"], out["n_free_days"], out["false_alarm_30d"] = 0, 0, np.nan
    return out


def tune_alarm(index, p, labels, leaks, mask=None, max_fa_30d=3.0,
               smooth_grid=(1, 3, 6, 12, 24), thr_grid=np.round(np.arange(0.3, 0.96, 0.05), 2),
               persist_grid=(0, 2, 6)):
    """
    Grid search of the alarm rule on validation predictions.
    Objective: most detected leaks subject to false_alarm_30d <= max_fa_30d; ties -> smaller median delay,
    then fewer false alarms. Returns (best params dict, full grid DataFrame).
    """
    rows = []
    for s in smooth_grid:
        for t in thr_grid:
            for k in persist_grid:
                a = alarm_rule(p, s, t, k)
                m = detector_metrics(index, p, a, labels, leaks, mask)
                rows.append(dict(smooth_h=s, thr=float(t), persist_h=k, **m))
    g = pd.DataFrame(rows)
    ok = g[g["false_alarm_30d"] <= max_fa_30d]
    if ok.empty:
        ok = g.nsmallest(1, "false_alarm_30d")
    best = ok.sort_values(["detected", "delay_h", "false_alarm_30d"], ascending=[False, True, True]).iloc[0]
    return {k: best[k] for k in ("smooth_h", "thr", "persist_h")}, g
