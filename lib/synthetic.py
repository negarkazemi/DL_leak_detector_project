"""
lib/synthetic.py -- synthetic leaks: model signature added to REAL leak-free sensor data.

Recipe (Hydraulic_Model_Guide, step 5):
  * take a leak-free stretch of the 2018 data (training period only)
  * pick a pipe, a diameter D ~ U(0.9, 2.4) cm and a shape (abrupt / incipient)
  * add  scale(t) * signature[pipe][hour(t)]  to the 33 pressures   (scale = (D/1.5cm)^2, size rule of part 1)
    and  scale(t) * extra_inflow[pipe]         to the inflow
  * label = cluster of that pipe
Real noise + simulated leak effect. Used ONLY for training; validation is always on real leaks.
"""
import os

import numpy as np
import pandas as pd

from .config import P1_DIR, REF_DIAMETER, STEPS_PER_HOUR


def load_pipe_library():
    """Pipe signatures from part 1: names, sig24 [905 x 24 x 33] (m, 1.5 cm leak), extra inflow [905] (m3/h), cluster."""
    S = np.load(os.path.join(P1_DIR, "pipe_signatures.npz"))
    N = np.load(os.path.join(P1_DIR, "node_signatures.npz"))
    from .simulator import load_network
    wn = load_network()
    ndq = dict(zip(N["nodes"], N["dq"]))
    dq = []
    for p in S["pipes"]:
        l = wn.get_link(p)
        ends = [ndq[n] for n in (l.start_node_name, l.end_node_name) if n in ndq]
        dq.append(np.mean(ends) if ends else 0.0)
    return dict(pipes=np.array(S["pipes"]), sig24=S["sig24"].astype(np.float32), dq=np.array(dq, np.float32),
                cluster=S["cluster"], weak=S["weak"])


def free_runs(leak_active, allowed):
    """Contiguous runs (start, end) of rows that are leak-free AND allowed (e.g. training rows)."""
    ok = (np.asarray(leak_active) == 0) & np.asarray(allowed)
    d = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def inject(S, labels, allowed, lib, rng, len_h=(24, 72), gap_h=12, warm_h=6, D_range=(0.009, 0.024),
           p_incipient=0.35):
    """
    One synthetic copy of the sensor data: episodes placed one after another inside every leak-free,
    allowed stretch. Returns (S_syn, y_cluster [n] (-1 = no synthetic leak), leak_flow [n] m3/h, episodes df).
    Pipes are drawn uniformly over DETECTABLE pipes (so clusters appear in proportion to their size).
    """
    P = S["P"].values.copy()
    F = S["F"].copy()
    hours = S["P"].index.hour.values
    n = len(P)
    ycl = np.full(n, -1, np.int16)
    flow = np.zeros(n, np.float32)
    cand = np.where(~lib["weak"])[0]
    eps = []
    for a, b in free_runs(labels["leak_active"].values, allowed):
        t = a + warm_h * STEPS_PER_HOUR                     # some clean history before the first episode
        while True:
            L = int(rng.uniform(*len_h) * STEPS_PER_HOUR)
            if t + L > b:
                break
            k = rng.choice(cand)
            D = rng.uniform(*D_range)
            full = (D / REF_DIAMETER) ** 2
            if rng.random() < p_incipient:                  # grows linearly, then constant
                ramp = max(1, int(L * rng.uniform(0.3, 0.8)))
                sc = full * np.minimum(1.0, (np.arange(L) + 1) / ramp)
                shape = "incipient"
            else:
                sc = np.full(L, full)
                shape = "abrupt"
            sl = slice(t, t + L)
            P[sl] += sc[:, None] * lib["sig24"][k][hours[sl]]
            F.iloc[sl, F.columns.get_loc("PUMP_1")] += sc * lib["dq"][k]
            ycl[sl] = lib["cluster"][k]
            flow[sl] = sc * lib["dq"][k]
            eps.append(dict(start=t, end=t + L, pipe=lib["pipes"][k], cluster=int(lib["cluster"][k]),
                            D_cm=round(D * 100, 2), shape=shape))
            t += L + gap_h * STEPS_PER_HOUR
    S2 = dict(S)
    S2["P"] = pd.DataFrame(P, index=S["P"].index, columns=S["P"].columns)
    S2["F"] = F
    return S2, ycl, flow, pd.DataFrame(eps)
