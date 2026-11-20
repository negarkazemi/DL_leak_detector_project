"""
lib/loc_features.py -- input of the localiser: the SHAPE of the pressure drop over the 33 sensors.

From the detector preprocessing H (pressure deviations in noise units z) we go back to metres
(multiply by the fitted residual scale), then for each averaging window A (hours):
    r_A(t)  = mean deviation over the last A hours                     [33]
    mode "change": r_A(t) - r_A(t - 7 days)  (change vs. the same hours one week earlier;
                   removes slow/seasonal bias of the normal-pressure model)
    centre  : subtract the mean over the 33 sensors   (the uniform drop says nothing about location)
    normalise to unit length                          (leak size removed, only shape kept)
Appended: log of the vector length (how strong the pattern is) and the unexplained-water z.
All causal.
"""
import numpy as np

from .config import STEPS_PER_HOUR

WEEK = 7 * 24 * STEPS_PER_HOUR


def pressure_metres(H, fb):
    pc = [c for c in H.columns if c.startswith("p_")]
    return H[pc].values * fb.params["scale"][: len(pc)][None, :]


def loc_inputs(H, fb, windows=(6, 24), mode="base"):
    import pandas as pd
    R = pd.DataFrame(pressure_metres(H, fb), index=H.index)
    parts, mags = [], []
    for A in windows:
        r = R.rolling(A * STEPS_PER_HOUR, min_periods=1).mean().values
        if mode == "change":
            prev = np.roll(r, WEEK, axis=0)
            prev[:WEEK] = 0.0
            r = r - prev
        r = r - r.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(r, axis=1, keepdims=True)
        parts.append(r / np.maximum(nrm, 1e-6))
        mags.append(np.log10(np.maximum(nrm, 1e-4)))
    w = H["water"].rolling(max(windows) * STEPS_PER_HOUR, min_periods=1).mean().values[:, None]
    return np.nan_to_num(np.column_stack(parts + mags + [w])).astype(np.float32)


def shape_vector(H, fb, A=24, mode="base"):
    """Only the centred, normalised A-hour pattern (for the nearest-signature baseline)."""
    return loc_inputs(H, fb, (A,), mode)[:, :H.columns.str.startswith("p_").sum()]
