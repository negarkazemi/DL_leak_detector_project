"""
part1/cluster.py -- Part 1, steps 3-7: pipe signatures -> low-effect pipes -> clusters -> clusters.csv

  3. pipe signature = mean of its end-node signatures                 [24 x 33]
  4. 33-vector per pipe (mean over the day), normalised to length 1
  5. sensor noise from REAL leak-free hours (train period only) -> SNR per pipe;
     pipes whose signal is below the noise go to their own "undetectable" cluster
  6. Ward clustering of the detectable pipes for every K in 10..20, pick K
  7. map the 20 real leaks to clusters, list clusters without a real example

Needs outputs/part1/node_signatures.npz (run `python -m part1.simulate` first).

usage:  python -m part1.cluster            # K chosen automatically (best silhouette)
        python -m part1.cluster --K 15     # force K (total number of clusters, incl. the weak one)
        python -m part1.cluster --snr 1.0 --snr-diameter 0.015

Outputs:
  clusters.csv                          pipe_id, cluster   (all 905 pipes, ids 0..K-1)
  outputs/part1/pipe_signatures.npz         pipe signatures + SNR (re-used for synthetic leaks later)
  outputs/part1/sensor_noise.csv            noise std per sensor
  outputs/part1/k_selection.csv             metrics for every K
  outputs/part1/cluster_summary.csv         size, #real leaks, ... per cluster
  outputs/part1/leak_to_cluster.csv         the 20 real leaks with their cluster
  outputs/part1/p1_*.png                      plots for the report
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score

from lib.config import CLUSTERS_CSV, P1_DIR as ART_DIR, P1_DIR as FIG_DIR, REF_DIAMETER, TRAIN_END
from lib.data import load_labels, load_leaks, load_pressures
from lib.simulator import load_network

K_RANGE = range(10, 21)          # allowed total number of clusters


# ------------------------------------------------------------------ step 3
def pipe_signatures(wn, node_sig):
    """Every pipe -> mean signature of its end junctions (tank/reservoir ends are skipped)."""
    names, sigs, n_ends = [], [], []
    for name in wn.pipe_name_list:
        pipe = wn.get_link(name)
        ends = [n for n in (pipe.start_node_name, pipe.end_node_name) if n in node_sig]
        names.append(name)
        n_ends.append(len(ends))
        sigs.append(np.mean([node_sig[n] for n in ends], axis=0) if ends
                    else np.zeros_like(next(iter(node_sig.values()))))
    return names, np.array(sigs), np.array(n_ends)


# ------------------------------------------------------------------ step 4/5 helpers
def remove_common_mode(A):
    """Subtract the mean over the 33 sensors (last axis).
    ~70% of every signature is a uniform drop of ALL sensors; it is the same for every pipe
    (says nothing about location) and looks exactly like normal day-to-day demand change."""
    return A - A.mean(axis=-1, keepdims=True)


def sensor_noise(sensors, centered):
    """
    Normal fluctuation of each sensor, from REAL data:
      - hourly means, only hours with no active leak, only the training period (no leakage)
      - remove the normal pattern (mean per hour-of-day x weekday/weekend)
      - optionally remove the common mode (same transform as the signatures)
      - hourly noise = std of hourly residuals
      - daily noise  = std of 24h-averaged residuals  (a real leak lasts days, so this is
                       the noise a sustained leak has to beat)
    """
    P = load_pressures()[sensors]
    y = load_labels()["leak_active"]
    Ph = P.resample("1h").mean()
    free = (y.resample("1h").max() == 0) & (Ph.index <= TRAIN_END)
    Pf = Ph[free]
    key = [Pf.index.hour, Pf.index.dayofweek >= 5]
    resid = Pf - Pf.groupby(key).transform("mean")
    if centered:
        resid = resid.sub(resid.mean(axis=1), axis=0)
    g = resid.groupby(resid.index.date)
    daily = g.mean()[g.size() >= 20]                 # days with >= 20 leak-free hours
    print(f"noise estimated from {free.sum()} leak-free training hours ({len(daily)} full days)")
    return resid.std(), daily.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None, help="total clusters (10..20); default = best silhouette")
    ap.add_argument("--snr", type=float, default=1.0, help="SNR threshold below which a pipe is 'undetectable'")
    ap.add_argument("--snr-diameter", type=float, default=REF_DIAMETER, help="leak size (m) at which SNR is judged")
    ap.add_argument("--raw", action="store_true", help="do NOT remove the common mode (for comparison)")
    args = ap.parse_args()
    centered = not args.raw

    S = np.load(os.path.join(ART_DIR, "node_signatures.npz"))
    sensors = list(S["sensors"])
    node_sig = dict(zip(S["nodes"], S["sig"]))
    assert len(node_sig) == 782, "node_signatures.npz is incomplete - run `python -m part1.simulate` without --limit"
    wn = load_network()

    # ---- step 3: pipe signatures
    names, sig24, n_ends = pipe_signatures(wn, node_sig)          # sig24: [905 x 24 x 33]
    X_raw = sig24.mean(axis=1)                                     # [905 x 33] mean over the day
    X = remove_common_mode(X_raw) if centered else X_raw            # feature used for clustering
    common_share = np.median(X_raw.mean(1) ** 2 * X_raw.shape[1] / np.maximum((X_raw ** 2).sum(1), 1e-12))
    print(f"median share of signature energy that is a uniform (common-mode) drop: {common_share:.2f}")

    # ---- step 4: normalise (shape, not size)
    norm = np.linalg.norm(X, axis=1)
    Xn = X / np.maximum(norm, 1e-12)[:, None]

    # ---- step 5: SNR and low-effect pipes
    noise_h, noise = sensor_noise(sensors, centered)               # noise = daily (sustained) noise
    pd.DataFrame({"noise_hourly_m": noise_h, "noise_daily_m": noise}).to_csv(os.path.join(ART_DIR, "sensor_noise.csv"))
    scale = (args.snr_diameter / REF_DIAMETER) ** 2               # size rule: effect ~ D^2
    snr = (np.abs(X) * scale / noise.values).max(axis=1)           # best sensor for each pipe
    snr_hourly = (np.abs(X) * scale / noise_h.values).max(axis=1)  # info only: visible within ONE hour?
    best_sensor = np.array(sensors)[(np.abs(X) / noise.values).argmax(axis=1)]
    weak = (snr < args.snr) | (n_ends == 0)
    strong = ~weak
    print(f"pipes: {len(names)},  undetectable (daily SNR<{args.snr} at D={args.snr_diameter * 100:.2f}cm): {weak.sum()}"
          f"   | not visible within a single hour: {(snr_hourly < args.snr).sum()}")

    # ---- step 6: Ward clustering of detectable pipes, every K
    Z = linkage(Xn[strong], method="ward")
    n_weak_cluster = 1 if weak.any() else 0
    rows, labels_by_K = [], {}
    for K in K_RANGE:
        k_strong = K - n_weak_cluster
        lab = fcluster(Z, k_strong, criterion="maxclust") - 1
        labels_by_K[K] = lab
        sizes = np.bincount(lab, minlength=k_strong)
        cent = np.array([Xn[strong][lab == c].mean(0) for c in range(k_strong)])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True)
        cc = cent @ cent.T
        np.fill_diagonal(cc, -1)
        # within-cluster cosine to own centroid
        within = np.mean([(Xn[strong][lab == c] @ cent[c]).mean() for c in range(k_strong)])
        # expected search effort if the right cluster is always ranked first: E[size/905]
        all_sizes = np.append(sizes, weak.sum()) if n_weak_cluster else sizes
        rows.append(dict(K=K,
                         silhouette_cos=silhouette_score(Xn[strong], lab, metric="cosine"),
                         within_cos=within,
                         max_between_centroid_cos=cc.max(),
                         largest_cluster=int(all_sizes.max()),
                         smallest_cluster=int(sizes.min()),
                         search_effort_ideal=float((all_sizes ** 2).sum() / len(names) ** 2)))
    ksel = pd.DataFrame(rows)
    ksel.to_csv(os.path.join(ART_DIR, "k_selection.csv"), index=False)
    print("\n=== K selection ===")
    print(ksel.round(3).to_string(index=False))

    K = args.K or int(ksel.loc[ksel["silhouette_cos"].idxmax(), "K"])
    assert K in K_RANGE, "K must be between 10 and 20"
    print(f"\nchosen K = {K}" + ("" if args.K else "  (best silhouette; override with --K)"))

    # final labels: detectable pipes 0..K-2 ordered by size (largest first), undetectable = K-1
    lab = labels_by_K[K]
    order = np.argsort(-np.bincount(lab))
    remap = {old: new for new, old in enumerate(order)}
    cluster = np.full(len(names), K - 1, dtype=int)
    cluster[strong] = [remap[c] for c in lab]

    clusters = pd.DataFrame({"pipe_id": names, "cluster": cluster})
    clusters.to_csv(CLUSTERS_CSV, index=False)
    print("saved", CLUSTERS_CSV)

    np.savez(os.path.join(ART_DIR, "pipe_signatures.npz"),
             pipes=np.array(names), sig24=sig24, mean33=X_raw, feature=X, centered=centered,
             snr=snr, snr_hourly=snr_hourly, weak=weak,
             cluster=cluster, sensors=np.array(sensors), diameter=REF_DIAMETER)

    # ---- step 7: real leaks -> clusters
    leaks = load_leaks()
    cl = clusters.set_index("pipe_id")["cluster"]
    leaks["cluster"] = leaks["pipe_id"].map(cl)
    pidx = {n: i for i, n in enumerate(names)}
    leaks["snr_at_own_D"] = [snr[pidx[p]] / scale * (d / REF_DIAMETER) ** 2
                             for p, d in zip(leaks["pipe_id"], leaks["diameter_m"])]
    leaks["undetectable_pipe"] = [bool(weak[pidx[p]]) for p in leaks["pipe_id"]]
    leaks[["leak_id", "pipe_id", "type", "diameter_m", "cluster", "snr_at_own_D", "undetectable_pipe"]] \
        .to_csv(os.path.join(ART_DIR, "leak_to_cluster.csv"), index=False)
    print("\n=== real leaks -> cluster ===")
    print(leaks[["leak_id", "pipe_id", "type", "diameter_m", "cluster", "snr_at_own_D"]].round(3).to_string(index=False))

    summ = clusters.groupby("cluster").size().rename("n_pipes").to_frame()
    summ["n_real_leaks"] = leaks.groupby("cluster").size().reindex(summ.index).fillna(0).astype(int)
    summ["real_leak_ids"] = leaks.groupby("cluster")["leak_id"].apply(",".join).reindex(summ.index).fillna("")
    summ["is_undetectable_cluster"] = (summ.index == K - 1) & weak.any()
    summ["median_snr"] = pd.Series(snr).groupby(cluster).median()
    summ.to_csv(os.path.join(ART_DIR, "cluster_summary.csv"))
    no_real = summ.index[summ["n_real_leaks"] == 0].tolist()
    print("\n=== clusters ===")
    print(summ.round(2).to_string())
    print("clusters WITHOUT a real 2018 leak (need synthetic leaks later):", no_real)

    # undetectable pipes: where are they? (for the report)
    ends_df = [f"{wn.get_link(n).start_node_name}-{wn.get_link(n).end_node_name}" for n in np.array(names)[weak]]
    weak_df = pd.DataFrame({"pipe_id": np.array(names)[weak], "nodes": ends_df, "snr": snr[weak],
                            "best_sensor": best_sensor[weak], "n_junction_ends": n_ends[weak]})
    weak_df.sort_values("snr").to_csv(os.path.join(ART_DIR, "undetectable_pipes.csv"), index=False)

    with open(os.path.join(ART_DIR, "part1_config.json"), "w") as f:
        json.dump(dict(K=K, centered=centered, snr_threshold=args.snr, snr_diameter=args.snr_diameter,
                       ref_diameter=REF_DIAMETER, n_undetectable=int(weak.sum()),
                       undetectable_cluster=int(K - 1) if weak.any() else None,
                       clusters_without_real_leak=[int(c) for c in no_real]), f, indent=2)

    # ------------------------------------------------------------------ figures
    make_figures(wn, ksel, K, Z, names, cluster, snr, X, noise, leaks, weak)


def make_figures(wn, ksel, K, Z, names, cluster, snr, X, noise, leaks, weak):
    # K selection
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ksel["K"], ksel["silhouette_cos"], "o-")
    ax[0].axvline(K, color="r", ls="--")
    ax[0].set(xlabel="K", ylabel="silhouette (cosine)", title="Cluster separation")
    ax[1].plot(ksel["K"], ksel["search_effort_ideal"], "o-", label="search effort (ideal)")
    ax[1].plot(ksel["K"], ksel["max_between_centroid_cos"], "s-", label="max centroid cosine")
    ax[1].axvline(K, color="r", ls="--")
    ax[1].set(xlabel="K", title="Trade-off")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p1_k_selection.png"), dpi=150)

    # dendrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(Z, truncate_mode="lastp", p=40, ax=ax, no_labels=True, color_threshold=Z[-(K - 2), 2])
    ax.set(title="Ward dendrogram of detectable pipes (normalised signatures)", ylabel="distance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p1_dendrogram.png"), dpi=150)

    # network map coloured by cluster, real leaks marked
    fig, ax = plt.subplots(figsize=(9, 8))
    wntr.graphics.plot_network(wn, link_attribute=pd.Series(cluster, index=names).astype(float),
                               link_cmap=plt.get_cmap("tab20", K), link_width=1.6,
                               node_size=0, ax=ax, add_colorbar=True, title=f"Pipe clusters (K={K})")
    for _, r in leaks.iterrows():
        link = wn.get_link(r["pipe_id"])
        a, b = link.start_node.coordinates, link.end_node.coordinates
        x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.plot(x, y, "k*", ms=10)
        ax.annotate(r["leak_id"], (x, y), fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, "p1_cluster_map.png"), dpi=150, bbox_inches="tight")

    # SNR histogram + detectable share vs diameter
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(np.log10(np.maximum(snr, 1e-3)), bins=50)
    ax[0].axvline(0, color="r", ls="--", label="SNR = 1")
    ax[0].set(xlabel="log10 daily SNR (best sensor, 1.5 cm leak)", ylabel="# pipes", title="Signal vs normal noise")
    ax[0].legend()
    Ds = np.linspace(0.006, 0.026, 21)
    base = (np.abs(X) / noise.values).max(axis=1)
    ax[1].plot(Ds * 100, [((base * (D / REF_DIAMETER) ** 2) >= 1).mean() * 100 for D in Ds], "o-")
    ax[1].axvspan(0.85, 2.4, alpha=0.15, label="2018 leak sizes")
    ax[1].set(xlabel="leak diameter (cm)", ylabel="% pipes with SNR >= 1", title="Detectable share vs size")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p1_snr.png"), dpi=150)

    # mean normalised signature of every cluster (heatmap over sensors)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1), 1e-12)[:, None]
    cent = np.array([Xn[(cluster == c) & ~weak].mean(0) if ((cluster == c) & ~weak).any()
                     else np.zeros(X.shape[1]) for c in range(K)])
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(cent, aspect="auto", cmap="viridis_r")
    ax.set(xlabel="sensor index", ylabel="cluster", title="Mean normalised signature per cluster")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p1_cluster_signatures.png"), dpi=150)
    plt.close("all")
    print("figures saved in", FIG_DIR)


if __name__ == "__main__":
    main()
