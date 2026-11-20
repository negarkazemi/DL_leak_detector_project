"""
lib/simulator.py -- running the hydraulic MODEL network (WNTR / EPANET) with or without a leak.
Used by part 1 (signatures) and later by the synthetic-leak generator.
"""
import os
import tempfile
import warnings

import numpy as np
import wntr

from .config import CD, G, INFLOWS, INP_FILE, KEEP, SIM_HOURS, STEP_S

warnings.filterwarnings("ignore")
_TMP = tempfile.mkdtemp(prefix="wntr_")   # EPANET temp files go here, not into the project folder


def emitter_coefficient(diameter_m):
    """Orifice q = Cd*A*sqrt(2 g p)  ->  EPANET emitter C (SI: m3/s per m^0.5)."""
    area = np.pi * (diameter_m / 2) ** 2
    return CD * area * np.sqrt(2 * G)


def load_network():
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    wn.options.time.duration = SIM_HOURS * 3600
    wn.options.time.hydraulic_timestep = STEP_S
    wn.options.time.report_timestep = STEP_S
    return wn


def run(wn, sensors):
    """Run EPANET once -> (sensor pressures [96 x 33] m, total inflow [96] m3/h), day 2 only."""
    r = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(_TMP, "sim"))
    p = r.node["pressure"][sensors].iloc[KEEP]
    q = r.link["flowrate"][INFLOWS].sum(axis=1).iloc[KEEP] * 3600.0   # m3/s -> m3/h
    return p, q


def simulate_node_leak(wn, node, diameter_m, sensors):
    """Put a leak (emitter) on one junction, simulate, then restore the network."""
    j = wn.get_node(node)
    old = j.emitter_coefficient
    j.emitter_coefficient = emitter_coefficient(diameter_m)
    try:
        return run(wn, sensors)
    finally:
        j.emitter_coefficient = old


def hour_of_day(index_seconds):
    """WNTR result index is seconds since start -> hour of day 0..23."""
    return (np.asarray(index_seconds) // 3600) % 24


def hourly_signature(dp):
    """[96 x 33] pressure difference -> [24 x 33] mean per hour of day."""
    return dp.groupby(hour_of_day(dp.index)).mean().values
