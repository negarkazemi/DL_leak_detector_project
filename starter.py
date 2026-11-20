"""
starter.py -- loading the data and simulating ONE leak on the model network.
Everything beyond this (all pipes, signatures, clusters, MLPs) is your job.

pip install wntr pandas numpy scikit-learn matplotlib   (or PyTorch for the MLPs)
"""
import numpy as np, pandas as pd, wntr

# ---------- 1. load the labelled year ----------
d = 'historical_2018'
read = lambda f: pd.read_csv(f'{d}/{f}', index_col=0, parse_dates=True)
P, F, L, D, y = read('pressures_m.csv'), read('flows_m3h.csv'), read('levels_m.csv'), read('demands_Lh.csv'), read('labels.csv')
leaks = pd.read_csv(f'{d}/leaks.csv', parse_dates=['start', 'peak', 'end'])
print(P.shape, 'share of time with a leak:', y['leak_active'].mean().round(2))
print(leaks[['leak_id', 'pipe_id', 'type', 'start', 'end', 'diameter_m']])

# ---------- 2. simulate the model network with and without one leak ----------
def simulate(emitter_node=None, diameter_m=0.015, hours=48):
    wn = wntr.network.WaterNetworkModel('network/L-TOWN_Model.inp')
    wn.options.time.duration = hours * 3600
    wn.options.time.hydraulic_timestep = 900
    wn.options.time.report_timestep = 900
    if emitter_node is not None:
        # orifice leak q = Cd * A * sqrt(2 g p)  ->  EPANET emitter coefficient (SI units, m3/s per m^0.5)
        wn.get_node(emitter_node).emitter_coefficient = 0.75 * np.pi * (diameter_m / 2) ** 2 * np.sqrt(2 * 9.81)
    r = wntr.sim.EpanetSimulator(wn).run_sim()
    return r.node['pressure'][P.columns]           # the 33 sensor junctions

base = simulate()
leak = simulate('n500')
delta = (leak - base).iloc[96:].mean()             # second day only (let the tank settle)
print('pressure change at the 33 sensors for a 1.5 cm leak at n500 (m):')
print(delta.round(3).sort_values().head(10))
