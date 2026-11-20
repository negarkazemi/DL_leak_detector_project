"""
part2.detector_experiments -- the detector comparisons asked for in part 2, one run each.
Every run appends a row to outputs/part2/detector/runs.csv; the summary table is printed at the end.

usage:  python -m part2.detector_experiments            (~20 min on a laptop CPU)
"""
import subprocess
import sys

import pandas as pd

from part2.detector import OUT

EXPERIMENTS = {
    "base":          "",                      # amr baseline, W = 12 h lags, MLP 128-64
    # --- non-neural baselines (same validation)
    "score_water":   "--model score --inputs water",
    "score_all":     "--model score --inputs all",
    "logreg":        "--model logreg",
    # --- raw pressure vs deviation from learned baseline
    "raw_none":      "--baseline none",
    "base_week":     "--baseline week",
    # --- effect of tank level and AMR separately (inside the normal-behaviour model)
    "base_nolevel":  "--baseline amr_nolevel",
    "base_noamr":    "--baseline level_noamr",
    # --- inputs: pressure only vs + inflow
    "only_pressure": "--inputs pressure",
    # --- window length 1..24 h
    "W1":            "--W 1",
    "W6":            "--W 6",
    "W24":           "--W 24",
    "W24_multi":     "--W 24 --mode multiscale",
    # --- MLP designs
    "mlp_small":     "--hidden 32",
    "mlp_deep":      "--hidden 256,128,64",
    # --- training label: ignore the first, tiny part of incipient leaks
    "ignore_small2": "--ignore-small 2",
    # --- training data: 20 real leaks vs + synthetic leaks (model signatures on leak-free hours)
    "syn_aug":       "--syn-copies 3",
    # --- simple time split for comparison
    "time_split":    "--split time",
}


def main():
    only = sys.argv[1:]
    for name, args in EXPERIMENTS.items():
        if only and name not in only:
            continue
        print(f"\n######## {name}: {args}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", "-m", "part2.detector", "--name", name, *args.split()],
                       check=True)
    runs = pd.read_csv(f"{OUT}/runs.csv").drop_duplicates("name", keep="last")
    cols = ["name", "n_inputs", "params", "AUROC", "fold_auroc", "detected", "n_leaks", "delay_h",
            "false_alarm_30d", "alarm_smooth_h", "alarm_thr", "alarm_persist_h", "train_s"]
    pd.set_option("display.width", 250)
    print(runs[cols].to_string(index=False))


if __name__ == "__main__":
    main()
