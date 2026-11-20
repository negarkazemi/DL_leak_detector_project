"""
part2.explore_features -- BEFORE training any network: which preprocessing makes leaks visible?

For every normal-behaviour baseline (none / daywe / week / amr):
  1. fit on leak-free TRAINING hours only, transform the whole year
  2. how well does it describe normal behaviour on unseen data?
       z-std and bias of the residual on leak-free VALIDATION hours
       (std ~1 and bias ~0 = the baseline generalises; larger = drift / seasonal error)
  3. how well does ONE simple score separate leak / no leak on validation (AUROC)?
       pressure score = minus the mean z over the 33 sensors (a leak lowers pressure)
       water score    = unexplained inflow z (a leak adds inflow)
     each as a 1-hour and a 24-hour causal mean
These single-feature AUROCs are also the simplest NON-neural baselines of part 2.

usage:  python -m part2.explore_features
Outputs: outputs/part2/feature_check.csv, outputs/part2/feature_timeline_<baseline>.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.config import P2_DIR, STEPS_PER_HOUR
from lib.data import load_labels, load_leaks, load_sensors, train_mask
from lib.features import BASELINES, FeatureBuilder


def scores(H):
    pcols = [c for c in H.columns if c.startswith("p_")]
    p1 = -H[pcols].mean(axis=1)
    w1 = H["water"]
    roll = lambda s: s.rolling(24 * STEPS_PER_HOUR, min_periods=1).mean()
    return {"pressure_1h": p1, "pressure_24h": roll(p1), "water_1h": w1, "water_24h": roll(w1)}


def main():
    S = load_sensors()
    y = load_labels()
    leaks = load_leaks()
    tr = train_mask(y.index)
    va = ~tr
    clean = y["leak_active"].rolling(STEPS_PER_HOUR, min_periods=1).max().values == 0
    print(f"train rows {tr.sum()}, val rows {va.sum()}, leak-free val rows {(va & clean).sum()} "
          f"(= {(va & clean).sum() / STEPS_PER_HOUR:.0f} h)")

    rows = []
    for b in BASELINES:
        fb = FeatureBuilder(b).fit(S, y, tr)
        H = fb.transform(S)
        pcols = [c for c in H.columns if c.startswith("p_")]
        r = {"baseline": b, "fit_rows": fb.params["n_fit_rows"]}
        for name, m in (("train", tr & clean), ("val", va & clean)):
            r[f"p_zstd_{name}_free"] = float(H.loc[m, pcols].std().mean())
            r[f"p_bias_{name}_free"] = float(H.loc[m, pcols].mean().mean())
            r[f"water_zstd_{name}_free"] = float(H.loc[m, "water"].std())
            r[f"water_bias_{name}_free"] = float(H.loc[m, "water"].mean())
        for sname, s in scores(H).items():
            r[f"AUROC_val_{sname}"] = roc_auc_score(y["leak_active"][va], s[va])
            r[f"AUROC_train_{sname}"] = roc_auc_score(y["leak_active"][tr], s[tr])
        rows.append(r)
        plot_timeline(H, y, leaks, b)
        print(f"done: {b}")

    res = pd.DataFrame(rows).set_index("baseline")
    res.to_csv(os.path.join(P2_DIR, "feature_check.csv"))
    pd.set_option("display.width", 200)
    print("\n=== how well does the baseline describe NORMAL behaviour (leak-free rows; ideal std~1, bias~0) ===")
    print(res[[c for c in res.columns if "zstd" in c or "bias" in c]].round(2).to_string())
    print("\n=== single-score AUROC (validation = Oct-Dec; train shown for comparison) ===")
    print(res[[c for c in res.columns if c.startswith("AUROC")]].round(3).to_string())


def plot_timeline(H, y, leaks, b):
    sc = scores(H)
    d = pd.DataFrame({"pressure (−mean z)": sc["pressure_1h"], "unexplained water (z)": sc["water_1h"]}).resample("1D").mean()
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for a, c, label in zip(ax, d.columns,
                           ("Pressure drop vs expected (z)", "Extra inlet flow vs expected (z)")):
        a.plot(d.index, d[c], color="tab:blue", lw=1.2)
        a.axhline(0, color="0.3", lw=0.6)
        a.set_ylabel(label)
        for _, l in leaks.iterrows():
            a.axvspan(l["start"], l["end"], color="tab:red", alpha=0.12)
        a.axvline(pd.Timestamp("2018-10-01"), color="tab:green", ls="--")
    for _, l in leaks.iterrows():
        ax[0].annotate(l["leak_id"], (l["start"], ax[0].get_ylim()[1]), fontsize=6, va="top")
    ax[0].legend(handles=[
        Line2D([0], [0], color="tab:blue", label="Daily mean of deviations"),
        Patch(color="tab:red", alpha=0.18, label="Recorded leak active"),
        Line2D([0], [0], color="tab:green", ls="--", label="Oct 1: exploratory split"),
    ], loc="upper right", fontsize=8)
    fig.suptitle(f"Deviation from expected normal behavior | method: {b}")
    ax[1].set_xlabel("Date in 2018")
    fig.tight_layout()
    fig.savefig(os.path.join(P2_DIR, f"feature_timeline_{b}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
