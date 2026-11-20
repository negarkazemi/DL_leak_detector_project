"""
part1/checks.py -- Part 1, validity checks of two modelling shortcuts (for the report).

  A. size rule:  does the signature scale like (D / 1.5cm)^2 ?
     For a few junctions and several diameters: fitted ratio vs predicted ratio,
     plus cosine to the 1.5 cm signature (is the SHAPE unchanged?).
  B. pipe = mean of its two end nodes ?
     For a few pipes: split the pipe in the middle (wntr.morph.split_pipe),
     put the leak on the new mid node, compare with the end-node average.

Needs outputs/part1/node_signatures.npz (run `python -m part1.simulate` first).
Outputs: outputs/part1/check_size_rule.csv, outputs/part1/check_pipe_split.csv,
         outputs/part1/p1_size_rule.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

from lib.config import P1_DIR as ART_DIR, P1_DIR as FIG_DIR, REF_DIAMETER
from lib.data import load_leaks, sensor_names
from lib.simulator import emitter_coefficient, hourly_signature, load_network, run, simulate_node_leak

DIAMETERS = [0.0085, 0.012, 0.015, 0.020, 0.024]   # covers the 2018 range 0.85-2.4 cm
N_SPLIT_PIPES = 10


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    S = np.load(os.path.join(ART_DIR, "node_signatures.npz"))
    nodes = list(S["nodes"])
    node_sig = dict(zip(nodes, S["sig"]))
    sensors = sensor_names()
    wn = load_network()
    p0, q0 = run(wn, sensors)

    # ------------------------------------------------ A. size rule
    leaks = load_leaks()
    test_nodes = ["n500"]
    for pid in leaks["pipe_id"][:5]:                       # end nodes of some real-leak pipes
        n = wn.get_link(pid).start_node_name
        if n in node_sig and n not in test_nodes:
            test_nodes.append(n)

    rows = []
    for n in test_nodes:
        ref = None
        for D in DIAMETERS:
            p1, q1 = simulate_node_leak(wn, n, D, sensors)
            s = hourly_signature(p1 - p0)
            if D == REF_DIAMETER:
                ref = s
            rows.append(dict(node=n, D_cm=D * 100, sig=s, dq=float((q1 - q0).mean())))
        for r in rows:
            if r["node"] == n:
                r["ratio_measured"] = float((r["sig"] * ref).sum() / (ref * ref).sum())   # least-squares scale
                r["ratio_predicted"] = (r["D_cm"] / (REF_DIAMETER * 100)) ** 2
                r["cos_to_ref"] = cos(r["sig"], ref)
    size = pd.DataFrame([{k: v for k, v in r.items() if k != "sig"} for r in rows])
    size["rel_error"] = (size["ratio_measured"] - size["ratio_predicted"]) / size["ratio_predicted"]
    size.to_csv(os.path.join(ART_DIR, "check_size_rule.csv"), index=False)
    print("\n=== A. size rule (signature ~ (D/1.5cm)^2) ===")
    print(size.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 4))
    for n, g in size.groupby("node"):
        ax.plot(g["ratio_predicted"], g["ratio_measured"], "o-", label=n)
    lim = [0, size[["ratio_predicted", "ratio_measured"]].values.max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("predicted ratio (D/1.5cm)^2")
    ax.set_ylabel("measured ratio (fit to 1.5 cm signature)")
    ax.set_title("Leak-size scaling rule")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p1_size_rule.png"), dpi=150)

    # ------------------------------------------------ B. pipe split vs mean of ends
    rng = np.random.default_rng(0)
    cand = [p for p in leaks["pipe_id"]] + list(rng.choice(wn.pipe_name_list, 30, replace=False))
    split_rows = []
    for pid in cand:
        if len(split_rows) >= N_SPLIT_PIPES:
            break
        pipe = wn.get_link(pid)
        ends = [n for n in (pipe.start_node_name, pipe.end_node_name) if n in node_sig]
        if len(ends) < 2:
            continue
        wn2 = wntr.morph.split_pipe(wn, pid, pid + "_B", pid + "_mid")
        wn2.get_node(pid + "_mid").emitter_coefficient = emitter_coefficient(REF_DIAMETER)
        p1, _ = run(wn2, sensors)
        s_split = hourly_signature(p1 - p0)
        s_mean = np.mean([node_sig[n] for n in ends], axis=0)
        split_rows.append(dict(pipe_id=pid, length_m=pipe.length,
                               cos_split_vs_mean=cos(s_split.mean(0), s_mean.mean(0)),
                               magnitude_ratio=float(np.linalg.norm(s_mean) / (np.linalg.norm(s_split) + 1e-12))))
    split = pd.DataFrame(split_rows)
    split.to_csv(os.path.join(ART_DIR, "check_pipe_split.csv"), index=False)
    print("\n=== B. leak at pipe middle vs mean of end nodes ===")
    print(split.round(4).to_string(index=False))
    print("mean cosine:", round(split["cos_split_vs_mean"].mean(), 4))


if __name__ == "__main__":
    main()
