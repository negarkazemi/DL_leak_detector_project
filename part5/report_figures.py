"""
part5.report_figures -- figures of the FINAL designs for the report (out-of-fold predictions).

  outputs/part5/fig_detector_output.png   p(t) and alarm over 2018 + zoom on one leak
  outputs/part5/fig_localizer_per_leak.png top-1 accuracy and search effort per leak, final vs baselines

usage:  python -m part5.report_figures
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.config import OUT_DIR, P2_DIR
from lib.data import load_labels, load_leaks
from lib.metrics import alarm_rule

P5 = os.path.join(OUT_DIR, "part5")
os.makedirs(P5, exist_ok=True)


def detector_figure():
    y = load_labels()
    idx = y.index
    leaks = load_leaks()
    cfg = json.load(open(os.path.join(P2_DIR, "detector", "final_detector.json")))
    p = pd.Series(np.load(os.path.join(P2_DIR, "detector", f"{cfg['design']}_ens_oof.npz"))["p"], index=idx)
    r = cfg["alarm_rule"]
    a = pd.Series(alarm_rule(p.values, r["smooth_h"], r["thr"], r["persist_h"]), index=idx)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6))
    for axi, (s, e), res in zip(ax, [("2018-01-01", "2018-12-31"), ("2018-02-08", "2018-02-26")], ["6h", "1h"]):
        am = a[s:e].resample(res).max()
        axi.plot(p[s:e].resample(res).mean(), lw=0.8, label="p(t) = leak probability")
        axi.fill_between(am.index, 0, am * 1.05, color="orange", alpha=0.35, step="post", label="alarm(t) = 1")
        for _, l in leaks.iterrows():
            if l["end"] >= pd.Timestamp(s) and l["start"] <= pd.Timestamp(e):
                axi.axvspan(max(l["start"], pd.Timestamp(s)), min(l["end"], pd.Timestamp(e)), color="r", alpha=0.12)
        axi.set_ylim(0, 1.08)
    ax[0].set_title("Final detector (ensemble, out-of-fold) over 2018 — red: real leak, orange: alarm")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[1].set_title("Zoom 8-26 Feb: leak L04 starts 14 Feb 18:10, repaired 21 Feb")
    fig.tight_layout()
    fig.savefig(os.path.join(P5, "fig_detector_output.png"), dpi=130)
    plt.close(fig)


def localizer_figure():
    d = os.path.join(P2_DIR, "localizer")
    fin = pd.read_csv(os.path.join(d, "L_copies12_ens_per_leak.csv"))
    near = pd.read_csv(os.path.join(d, "nearest_per_leak.csv"))
    syn = pd.read_csv(os.path.join(d, "L_syn_ens_per_leak.csv"))
    x = np.arange(len(fin))
    w = 0.27
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for k, (col, lab) in enumerate([("top1", "top-1 accuracy"), ("search_effort", "search effort (share of 905 pipes)")]):
        ax[k].bar(x - w, near[col], w, label="nearest signature (non-neural)", color="0.7")
        ax[k].bar(x, syn[col], w, label="MLP, synthetic only", color="C1")
        ax[k].bar(x + w, fin[col], w, label="MLP, real + synthetic (final)", color="C0")
        ax[k].set_ylabel(lab)
        ax[k].grid(axis="y", alpha=0.3)
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(fin["leak_id"])
    ax[0].set_title("Localiser per real leak (ensembles, out-of-fold, alarm steps of the final detector)")
    fig.tight_layout()
    fig.savefig(os.path.join(P5, "fig_localizer_per_leak.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    detector_figure()
    localizer_figure()
    print("saved to", P5)
