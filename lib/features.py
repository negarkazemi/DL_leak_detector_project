"""
lib/features.py -- preprocessing shared by training AND predict.py.

Two strictly separated phases:
    fb = FeatureBuilder(baseline="amr").fit(sensors, labels, train_mask)   # ONLY on training data
    fb.save(path)                                                         # -> models/ (deliverable)
    H  = FeatureBuilder.load(path).transform(sensors)                     # on ANY data, no labels needed

transform() returns one row per 5-min timestamp ("hourly features"):
    p_<sensor>   33 cols  pressure deviation from the learned normal pressure, in units of normal noise (z)
    water        1 col    unexplained inflow: inflow - expected inflow, z units
    hour_sin/cos, dow_sin/cos, weekend  time of day / day of week
Every value at time t uses only data at t and BEFORE (causal 1-hour rolling mean).

make_window(H, W, mode) then turns these into the fixed-length MLP input x(t).

Normal-behaviour models ("baseline"), compared in part 2:
    "none"   raw signals, only standardised            (the "raw pressure" variant)
    "daywe"  mean per hour-of-day x {weekday, weekend}
    "week"   mean per hour-of-week (168 slots)
    "amr"    linear regression on hour-of-day, weekend, AMR demand of 82 homes, tank level
             -> explains demand-driven pressure changes (hot days, holidays, seasons)
    "amr_nolevel" / "level_noamr"   the same regression without tank level / without AMR
             (to measure the effect of each signal separately)
All baselines are fitted ONLY on leak-free training timestamps.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import INFLOWS, STEPS_PER_HOUR, TANK

BASELINES = ("none", "daywe", "week", "amr", "amr_nolevel", "level_noamr")
# regression baselines: which explanatory signals they use (AMR demand, tank level)
REGRESSION = {"amr": (True, True), "amr_nolevel": (True, False), "level_noamr": (False, True)}


# ------------------------------------------------------------------ helpers
def _hourly(df):
    """Causal 1-hour mean: value at t = mean of the last 12 five-minute samples (t included)."""
    return df.rolling(STEPS_PER_HOUR, min_periods=1).mean()


def _signals(S):
    """Raw sensor tables -> hourly-smoothed signals used by every baseline."""
    P = _hourly(S["P"])
    inflow = _hourly(S["F"][INFLOWS].sum(axis=1)).rename("inflow")
    level = _hourly(S["L"][TANK]).rename("level")
    amr = _hourly(S["D"].sum(axis=1) / 1000.0).rename("amr")          # 82 homes, L/h -> m3/h
    dlevel = level.diff(STEPS_PER_HOUR).fillna(0.0).rename("dlevel")  # tank filling(+)/emptying(-) over last hour
    return P, inflow, level, amr, dlevel


TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend"]


def _time_feats(idx):
    """Hour of day and day of week (brief: 'in all cases add hour of day and day of week'), cyclic."""
    h = idx.hour + idx.minute / 60.0
    d = idx.dayofweek + h / 24.0                      # continuous day of week, Monday = 0
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                         "dow_sin": np.sin(2 * np.pi * d / 7), "dow_cos": np.cos(2 * np.pi * d / 7),
                         "weekend": (idx.dayofweek >= 5).astype(float)}, index=idx)


def _design(idx, amr, level, dlevel, use_amr=True, use_level=True):
    """Regression inputs: one-hot hour x weekend + (AMR demand) + (tank state)."""
    hw = (idx.hour + 24 * (idx.dayofweek >= 5)).values
    cols = [np.eye(48)[hw]]
    if use_amr:
        cols += [amr.values, amr.values ** 2]
    if use_level:
        cols += [level.values, dlevel.values]
    return np.column_stack(cols)


class FeatureBuilder:
    def __init__(self, baseline="amr"):
        assert baseline in BASELINES, baseline
        self.baseline = baseline
        self.params = {}

    # -------------------------------------------------------------- fit (training data only)
    def fit(self, S, labels, mask):
        """S: dict from lib.data.load_sensors; labels: labels.csv; mask: bool array = training rows."""
        P, inflow, level, amr, dlevel = _signals(S)
        # a timestamp is "leak-free" only if no leak was active during its whole 1-hour smoothing window
        clean = labels["leak_active"].rolling(STEPS_PER_HOUR, min_periods=1).max().values == 0
        fit_rows = np.asarray(mask) & clean
        self.params["n_fit_rows"] = int(fit_rows.sum())
        self.params["sensors"] = list(S["P"].columns)
        Y = pd.concat([P, inflow], axis=1)          # 33 pressures + inflow, modelled the same way

        if self.baseline == "none":
            self.params["mean"] = Y[np.asarray(mask)].mean().values
        elif self.baseline in ("daywe", "week"):
            key = self._key(Y.index)
            tab = Y[fit_rows].groupby(key[fit_rows]).mean()
            self.params["table"] = tab                                  # slots x 34
            # fallback for a slot never seen leak-free: mean per hour-of-day
            self.params["fallback"] = Y[fit_rows].groupby(Y.index[fit_rows].hour).mean()
        else:  # regression baselines
            Xd = _design(Y.index, amr, level, dlevel, *REGRESSION[self.baseline])
            reg = Ridge(alpha=1.0).fit(Xd[fit_rows], Y.values[fit_rows])
            self.params["reg"] = reg

        R = Y.values - self._expected(Y.index, amr, level, dlevel)
        self.params["scale"] = R[fit_rows].std(axis=0) if self.baseline != "none" else Y[np.asarray(mask)].std().values
        return self

    # -------------------------------------------------------------- transform (any data)
    def transform(self, S):
        assert self.params, "call fit() or load() first"
        assert list(S["P"].columns) == self.params["sensors"], "sensor columns differ from training"
        P, inflow, level, amr, dlevel = _signals(S)
        Y = pd.concat([P, inflow], axis=1)
        R = (Y.values - self._expected(Y.index, amr, level, dlevel)) / self.params["scale"]
        cols = [f"p_{s}" for s in self.params["sensors"]] + ["water"]
        H = pd.DataFrame(R, index=Y.index, columns=cols).astype(np.float32)
        return pd.concat([H, _time_feats(Y.index).astype(np.float32)], axis=1)

    # -------------------------------------------------------------- internals
    def _key(self, idx):
        if self.baseline == "week":
            return np.asarray(idx.dayofweek * 24 + idx.hour)
        return np.asarray(idx.hour + 24 * (idx.dayofweek >= 5))

    def _expected(self, idx, amr, level, dlevel):
        """Normal (leak-free) value of the 33 pressures + inflow at every timestamp."""
        if self.baseline == "none":
            return self.params["mean"][None, :]
        if self.baseline in REGRESSION:
            return self.params["reg"].predict(_design(idx, amr, level, dlevel, *REGRESSION[self.baseline]))
        tab, fb = self.params["table"], self.params["fallback"]
        E = tab.reindex(self._key(idx)).values
        miss = np.isnan(E).any(axis=1)
        if miss.any():
            E[miss] = fb.reindex(idx.hour[miss]).values
        return E

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"baseline": self.baseline, "params": self.params}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        fb = cls(d["baseline"])
        fb.params = d["params"]
        return fb


# ------------------------------------------------------------------ window -> MLP input
SCALES_H = (1, 3, 6, 12, 24)


def make_window(H, W_hours=12, mode="lags", cols=None):
    """
    Fixed-length input x(t) from the hourly features H (output of transform), causal.

    mode="lags":       values at t, t-1h, ..., t-(W-1)h        -> len(cols) * W numbers
    mode="multiscale": mean over the last 1,3,6,12,24 h (<= W)   -> len(cols) * len(scales) numbers
    Time features (hour and day of week, weekend) are appended once, at time t.
    Start of the file (window not yet full): missing past = 0 = "normal" (no future data used).
    """
    cols = cols or [c for c in H.columns if c.startswith("p_")] + ["water"]
    base = H[cols]
    parts, names = [], []
    if mode == "lags":
        for k in range(W_hours):
            parts.append(base.shift(k * STEPS_PER_HOUR).values)
            names += [f"{c}@-{k}h" for c in cols]
    elif mode == "multiscale":
        for h in [s for s in SCALES_H if s <= W_hours]:
            # H is already a 1-hour mean; average it over the last h hours (causal)
            parts.append(base.rolling(h * STEPS_PER_HOUR, min_periods=1).mean().values)
            names += [f"{c}_mean{h}h" for c in cols]
    else:
        raise ValueError(mode)
    tf = H[TIME_COLS].values
    X = np.nan_to_num(np.column_stack(parts + [tf]), nan=0.0).astype(np.float32)
    return X, names + TIME_COLS
