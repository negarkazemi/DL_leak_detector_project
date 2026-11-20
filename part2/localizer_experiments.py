"""
part2.localizer_experiments -- the localiser comparisons of part 2 (one run each, seed 0).
Each run appends a row to outputs/part2/localizer/runs.csv; summary printed at the end.

usage:  python -m part2.localizer_experiments [names...]     (~8 min)
"""
import subprocess
import sys

import pandas as pd

from part2.localizer import OUT

EXPERIMENTS = {
    # --- baselines without any learning
    "prior":          "--model prior",
    "nearest":        "--model nearest",
    "nearest_change": "--model nearest --mode change",
    # --- training data: real / real + synthetic / synthetic only
    "real":           "--data real",
    "both":           "--data both",
    "syn":            "--data syn",
    # --- input: change vs one week earlier instead of deviation from normal model
    "both_change":    "--data both --mode change",
    # --- averaging windows
    "both_w24":       "--data both --windows 24",
    "both_w1-6-24":   "--data both --windows 1,6,24",
    # --- MLP designs
    "both_small":     "--data both --hidden 32",
    "both_deep":      "--data both --hidden 256,128,64",
    # --- training choices
    "both_cw":        "--data both --class-weight",
    "both_copies12":  "--data both --copies 12",
    "both_minflow5":  "--data both --min-flow 5",
}


def main():
    only = sys.argv[1:]
    for name, args in EXPERIMENTS.items():
        if only and name not in only:
            continue
        print(f"\n######## {name}: {args}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", "-m", "part2.localizer", "--name", name, *args.split()],
                       check=True, stdout=subprocess.DEVNULL)
    runs = pd.read_csv(f"{OUT}/runs.csv").drop_duplicates("name", keep="last")
    runs = runs[runs["name"].isin(EXPERIMENTS)]
    cols = ["name", "n_inputs", "params", "n_real_train", "n_syn_train", "search_effort", "top1",
            "search_effort_oracle", "top1_oracle", "train_s"]
    pd.set_option("display.width", 250)
    print(runs[cols].to_string(index=False))


if __name__ == "__main__":
    main()
