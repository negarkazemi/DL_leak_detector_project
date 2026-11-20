"""
lib/data.py -- loading a data folder (the 2018 folder OR the instructor's evaluation folder).

The evaluation folder has the same sensor files but NO labels.csv / leaks.csv /
leak_flows_m3h.csv, so everything that needs labels is optional here.
"""
import os

import pandas as pd

import numpy as np

from .config import DATA_DIR, FOLD_EDGES, PURGE_HOURS, TRAIN_END


def _read(folder, name):
    return pd.read_csv(os.path.join(folder, name), index_col=0, parse_dates=True)


def load_sensors(folder=DATA_DIR):
    """dict with the four sensor tables, all on the same 5-min index:
       P pressures (33 cols, m), F flows (3 cols, m3/h), L tank level (1 col, m), D AMR demands (82 cols, L/h)"""
    d = {"P": _read(folder, "pressures_m.csv"),
         "F": _read(folder, "flows_m3h.csv"),
         "L": _read(folder, "levels_m.csv"),
         "D": _read(folder, "demands_Lh.csv")}
    idx = d["P"].index
    for k, v in d.items():
        if not v.index.equals(idx):             # tolerate small misalignments: align on the pressure clock
            print(f"warning: {k} has a different time index than pressures -> aligned (forward fill)")
            d[k] = v.reindex(idx, method="ffill")
        # missing values: carry the last known value forward (never uses future data);
        # anything still missing at the very start stays NaN and is treated as "normal" later
        d[k] = d[k].ffill()
    return d


def load_labels(folder=DATA_DIR):
    return _read(folder, "labels.csv")


def load_leaks(folder=DATA_DIR):
    return pd.read_csv(os.path.join(folder, "leaks.csv"), parse_dates=["start", "peak", "end"])


def load_pressures(folder=DATA_DIR):
    return _read(folder, "pressures_m.csv")


def sensor_names(folder=DATA_DIR):
    """The 33 pressure-sensor junctions, in file column order."""
    return list(pd.read_csv(os.path.join(folder, "pressures_m.csv"), nrows=0).columns[1:])


def train_mask(index):
    """True for timestamps in the training period (Jan-Sep 2018)."""
    return index <= pd.Timestamp(TRAIN_END)


def splits(index, kind="cv"):
    """
    List of (name, train_mask, val_mask).
      kind="cv":   4 event-aligned time blocks; each block is validation once, the rest is training.
                   Training rows within PURGE_HOURS of the validation block are dropped, so no
                   training window overlaps validation data.
      kind="time": a single split, train Jan-Sep / validation Oct-Dec.
    """
    if kind == "time":
        tr = np.asarray(train_mask(index))
        return [("time", tr, ~tr)]
    out = []
    purge = pd.Timedelta(hours=PURGE_HOURS)
    for k in range(len(FOLD_EDGES) - 1):
        a, b = pd.Timestamp(FOLD_EDGES[k]), pd.Timestamp(FOLD_EDGES[k + 1])
        va = np.asarray((index >= a) & (index < b))
        near = np.asarray((index >= a - purge) & (index < b + purge))
        out.append((f"fold{k}", ~near, va))
    return out
