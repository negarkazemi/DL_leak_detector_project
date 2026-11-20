"""
part1.simulate -- Part 1, steps 1-2: leak signature of every junction.

  1. simulate the model network WITHOUT a leak (baseline p0, q0)
  2. for each of the 782 junctions: put a 1.5 cm leak there, simulate,
     signature = p1 - p0 averaged per hour of day  ->  [24 x 33]
     extra inflow = mean(q1 - q0)                  ->  scalar (m3/h)

Output: outputs/part1/node_signatures.npz  (run once, ~4-10 min; later scripts only load it)

usage:  python -m part1.simulate            # all junctions
        python -m part1.simulate --limit 20 # quick test
"""
import argparse
import os
import time

import numpy as np

from lib.config import P1_DIR as ART_DIR, REF_DIAMETER
from lib.data import sensor_names
from lib.simulator import hourly_signature, load_network, run, simulate_node_leak

OUT = os.path.join(ART_DIR, "node_signatures.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N junctions (testing)")
    ap.add_argument("--diameter", type=float, default=REF_DIAMETER)
    args = ap.parse_args()

    sensors = sensor_names()
    wn = load_network()

    # ---- step 1: baseline, no leak
    p0, q0 = run(wn, sensors)
    base_hourly = hourly_signature(p0)                 # [24 x 33] normal model pressure (for reference only)

    # ---- step 2: one leak per junction
    nodes = wn.junction_name_list[: args.limit]
    sig = np.zeros((len(nodes), 24, len(sensors)), dtype=np.float32)
    dq = np.zeros(len(nodes), dtype=np.float32)
    t0 = time.time()
    for i, j in enumerate(nodes):
        p1, q1 = simulate_node_leak(wn, j, args.diameter, sensors)
        sig[i] = hourly_signature(p1 - p0)
        dq[i] = (q1 - q0).mean()
        if (i + 1) % 50 == 0 or i + 1 == len(nodes):
            el = time.time() - t0
            print(f"{i + 1}/{len(nodes)} junctions  {el:.0f}s elapsed, ~{el / (i + 1) * (len(nodes) - i - 1):.0f}s left")

    np.savez(OUT, nodes=np.array(nodes), sensors=np.array(sensors), sig=sig, dq=dq,
             base_hourly=base_hourly, diameter=args.diameter)
    print("saved", OUT)

    k = nodes.index("n500") if "n500" in nodes else 0
    print(f"sanity check {nodes[k]}: extra inflow {dq[k]:.2f} m3/h (guide: ~15 for n500), "
          f"largest pressure drop {sig[k].mean(0).min():.3f} m")


if __name__ == "__main__":
    main()
