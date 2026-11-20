"""
part4.hypothesis -- Part 4: test of the riskiest assumption.

ASSUMPTION
  The leak signatures simulated on the hydraulic MODEL (part 1) transfer to the REAL network:
  a real leak on pipe X changes the 33 pressures in (approximately) the same spatial pattern as
  the simulated leak on pipe X. Clusters, synthetic leaks and the localiser all rest on this.

PRE-REGISTERED SUCCESS CRITERIA (fixed before test 1 was run)
  Test 1 (direct): for every real leak, measured pressure change (leak hours minus surrounding
     leak-free hours, matched by hour of day) vs the simulated signature of the same pipe.
     H1 holds if the MEDIAN centred cosine over the usable leaks is >= 0.7.
     Also reported: raw cosine, magnitude ratio measured/simulated (at the real leak size), and the
     rank of the true pipe / true cluster among all candidates when matched by cosine.
  Test 2 (indirect): localiser trained ONLY on synthetic leaks, evaluated on the 20 real leaks.
     H2 holds if its search effort is below the non-neural nearest-signature baseline (0.232) and at
     most 0.05 worse than the real+synthetic localiser.
     NOTE: test 2 was run in part 2 (design L_syn), so its numbers were seen before these criteria
     were written down; only test 1 is a truly blind test.

usage:  python -m part4.hypothesis
Outputs (outputs/part4/): test1_per_leak.csv, test2_per_leak.csv, summary.json, *.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.config import CLUSTERS_CSV, DATA_DIR, P2_DIR, P4_DIR, REF_DIAMETER
from lib.data import load_labels, load_leaks, load_pressures
from lib.synthetic import load_pipe_library

COS_THRESHOLD = 0.7
WINDOW_D = (14, 45)    # reference leak-free hours within +- 14 days of the leak; if too few, +- 45 days
MIN_HOURS_OF_DAY = 6
OTHER_MAX = 0.15       # other simultaneous leaks may carry at most 15 % of this leak's flow   # need at least this many hours of day present in both leak and reference


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def centre(v):
    return v - v.mean(axis=-1, keepdims=True)


def test1():
    P = load_pressures()
    y = load_labels()
    leaks = load_leaks()
    LF = pd.read_csv(os.path.join(DATA_DIR, "leak_flows_m3h.csv"), index_col=0, parse_dates=True)
    lib = load_pipe_library()
    pidx = {p: i for i, p in enumerate(lib["pipes"])}
    clusters = pd.read_csv(CLUSTERS_CSV).set_index("pipe_id")["cluster"]

    Ph = P.resample("1h").mean()
    LFh = LF.resample("1h").mean()
    any_leak = y["leak_active"].resample("1h").max() > 0
    after_repair = pd.Series(False, index=Ph.index)
    for _, l in leaks.iterrows():
        after_repair |= (Ph.index > l["end"]) & (Ph.index <= l["end"] + pd.Timedelta(hours=24))
    free = ~any_leak & ~after_repair

    # signature library for ranking: centred, unit length, detectable pipes only
    det = ~lib["weak"]
    lib_cent = centre(lib["sig24"].mean(axis=1))
    lib_cent /= np.maximum(np.linalg.norm(lib_cent, axis=1, keepdims=True), 1e-12)
    K = int(clusters.max()) + 1
    cent_c = np.zeros((K, lib_cent.shape[1]))
    for c in range(K):
        m = (lib["cluster"] == c) & det
        if m.any():
            cent_c[c] = lib_cent[m].mean(0) / np.linalg.norm(lib_cent[m].mean(0))

    rows, examples = [], {}
    for _, l in leaks.iterrows():
        pid = l["pipe_id"]
        f = LFh[pid]
        others = LFh.drop(columns=pid).sum(axis=1)
        # this leak at (nearly) full size, other leaks absent or negligible (<= OTHER_MAX of its flow;
        # needed because the tiny leak L13 overlaps everything from mid-Sep to end of Nov)
        full = (f >= 0.9 * f.max()) & (others <= OTHER_MAX * f)
        for wd in WINDOW_D:
            near = (Ph.index >= l["start"] - pd.Timedelta(days=wd)) & (Ph.index <= l["end"] + pd.Timedelta(days=wd))
            ref = free & near
            hours = sorted(set(Ph.index[full].hour) & set(Ph.index[ref].hour))
            if len(hours) >= MIN_HOURS_OF_DAY:
                break
        r = dict(leak_id=l["leak_id"], pipe=pid, type=l["type"], D_cm=round(l["diameter_m"] * 100, 2),
                 cluster=int(clusters[pid]), n_leak_h=int(full.sum()), n_ref_h=int(ref.sum()), n_hours_of_day=len(hours),
                 ref_window_d=wd)
        if len(hours) < MIN_HOURS_OF_DAY:
            r["usable"] = False
            rows.append(r)
            continue
        A = Ph[full].groupby(Ph.index[full].hour).mean().loc[hours]
        B = Ph[ref].groupby(Ph.index[ref].hour).mean().loc[hours]
        meas = (A - B).values.mean(axis=0)                            # [33] measured change, m
        k = pidx[pid]
        sim = lib["sig24"][k][hours].mean(axis=0) * (l["diameter_m"] / REF_DIAMETER) ** 2   # at the real size
        mc, sc = centre(meas), centre(sim)
        sims_all = lib_cent @ (mc / max(np.linalg.norm(mc), 1e-12))
        sims_all[~det] = -np.inf
        rank_pipe = int((sims_all > sims_all[k]).sum() + 1)
        cl_sims = cent_c @ (mc / max(np.linalg.norm(mc), 1e-12))
        rank_cl = int((cl_sims > cl_sims[r["cluster"]]).sum() + 1)
        r.update(usable=True, cos_raw=cos(meas, sim), cos_centred=cos(mc, sc),
                 ratio_raw=float(np.linalg.norm(meas) / np.linalg.norm(sim)),
                 ratio_centred=float(np.linalg.norm(mc) / np.linalg.norm(sc)),
                 mean_drop_meas_m=float(-meas.mean()), mean_drop_sim_m=float(-sim.mean()),
                 rank_true_pipe_of_900=rank_pipe, rank_true_cluster_of_20=rank_cl,
                 best_cluster=int(np.argmax(cl_sims)))
        rows.append(r)
        examples[l["leak_id"]] = (mc, sc)
    return pd.DataFrame(rows), examples


def test2():
    d = os.path.join(P2_DIR, "localizer")
    syn = pd.read_csv(os.path.join(d, "L_syn_ens_per_leak.csv"))
    both = pd.read_csv(os.path.join(d, "L_copies12_ens_per_leak.csv"))
    near = pd.read_csv(os.path.join(d, "nearest_per_leak.csv"))
    t = syn[["leak_id", "cluster", "search_effort", "top1", "mean_rank"]].rename(
        columns={"search_effort": "se_syn_only", "top1": "top1_syn_only", "mean_rank": "rank_syn_only"})
    t = t.merge(both[["leak_id", "search_effort", "top1"]].rename(
        columns={"search_effort": "se_real_syn", "top1": "top1_real_syn"}), on="leak_id")
    t = t.merge(near[["leak_id", "search_effort", "top1"]].rename(
        columns={"search_effort": "se_nearest", "top1": "top1_nearest"}), on="leak_id")
    S = pd.read_csv(os.path.join(d, "seeds_summary.csv")).set_index("design")
    summ = {"se_syn_only_ens": float(S.loc["L_syn", "ens_search_effort"]),
            "top1_syn_only_ens": float(S.loc["L_syn", "ens_top1"]),
            "se_real_syn_ens": float(S.loc["L_copies12", "ens_search_effort"]),
            "top1_real_syn_ens": float(S.loc["L_copies12", "ens_top1"]),
            "se_syn_only_single": [float(S.loc["L_syn", "search_effort_mean"]), float(S.loc["L_syn", "search_effort_std"])],
            "se_nearest": float(near["search_effort"].mean()), "top1_nearest": float(near["top1"].mean())}
    return t, summ


def main():
    t1, ex = test1()
    t2, s2 = test2()
    u = t1[t1["usable"]]
    med = float(u["cos_centred"].median())
    s1 = {"n_usable": int(len(u)), "median_cos_centred": med, "median_cos_raw": float(u["cos_raw"].median()),
          "share_cos_centred_ge_threshold": float((u["cos_centred"] >= COS_THRESHOLD).mean()),
          "median_ratio_raw": float(u["ratio_raw"].median()),
          "median_rank_true_cluster": float(u["rank_true_cluster_of_20"].median()),
          "share_true_cluster_top1": float((u["rank_true_cluster_of_20"] == 1).mean()),
          "share_true_cluster_top3": float((u["rank_true_cluster_of_20"] <= 3).mean()),
          "H1_holds": med >= COS_THRESHOLD}
    s2["H2_holds"] = (s2["se_syn_only_ens"] < 0.232) and (s2["se_syn_only_ens"] <= s2["se_real_syn_ens"] + 0.05)
    t = t2.merge(t1[["leak_id", "cos_centred", "ratio_raw"]], on="leak_id", how="left")
    t1.to_csv(os.path.join(P4_DIR, "test1_per_leak.csv"), index=False)
    t.to_csv(os.path.join(P4_DIR, "test2_per_leak.csv"), index=False)
    with open(os.path.join(P4_DIR, "summary.json"), "w") as f:
        json.dump({"test1": s1, "test2": s2, "criteria": {"cos_threshold": COS_THRESHOLD}}, f, indent=2)

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    col = ["C0" if ty == "abrupt" else "C1" for ty in u["type"]]
    ax[0].bar(u["leak_id"], u["cos_centred"], color=col)
    ax[0].axhline(COS_THRESHOLD, color="r", ls="--", lw=1)
    ax[0].set(ylabel="cosine (centred)", title="Test 1: measured vs simulated pattern (blue=abrupt, orange=incipient)")
    ax[0].tick_params(axis="x", rotation=90)
    ax[1].scatter(u["D_cm"], u["ratio_raw"], c=col)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set(xlabel="leak diameter (cm)", ylabel="|measured| / |simulated|", title="Magnitude ratio at real leak size")
    fig.tight_layout()
    fig.savefig(os.path.join(P4_DIR, "test1_cosine.png"), dpi=140)
    best = u.sort_values("cos_centred")
    pick = [best.iloc[-1]["leak_id"], best.iloc[len(best) // 2]["leak_id"], best.iloc[0]["leak_id"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, lid in zip(ax, pick):
        mc, sc = ex[lid]
        a.plot(mc, "o-", ms=3, label="measured (centred)")
        a.plot(sc, "s-", ms=3, label="simulated (centred)")
        a.set(title=f"{lid}: cos={u.set_index('leak_id').loc[lid, 'cos_centred']:.2f}", xlabel="sensor index", ylabel="m")
        a.legend(fontsize=7)
    fig.suptitle("Best / median / worst agreement")
    fig.tight_layout()
    fig.savefig(os.path.join(P4_DIR, "test1_examples.png"), dpi=140)

    pd.set_option("display.width", 220)
    print("=== TEST 1: measured vs simulated signature, per leak ===")
    print(t1.round(3).to_string(index=False))
    print(json.dumps(s1, indent=2))
    print("\n=== TEST 2: synthetic-only localiser vs real+synthetic vs nearest ===")
    print(t.round(3).to_string(index=False))
    print(json.dumps(s2, indent=2))


if __name__ == "__main__":
    main()
