"""
lib/config.py -- every path and constant of the project, in ONE place.
"""
import os

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "historical_2018")        # given labelled year
INP_FILE = os.path.join(ROOT, "network", "L-TOWN_Model.inp")
CLUSTERS_CSV = os.path.join(ROOT, "clusters.csv")       # deliverable of part 1
MODELS_DIR = os.path.join(ROOT, "models")               # final weights + preprocessing (deliverable)
OUT_DIR = os.path.join(ROOT, "outputs")                 # everything produced by experiments
P1_DIR = os.path.join(OUT_DIR, "part1")
P2_DIR = os.path.join(OUT_DIR, "part2")
P4_DIR = os.path.join(OUT_DIR, "part4")
for _d in (MODELS_DIR, P1_DIR, P2_DIR, P4_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- network / sensors
INFLOWS = ["PUMP_1", "p227", "p235"]   # links whose flow = all water entering the network
TANK = "T1"

# ---------------------------------------------------------------- simulation (part 1)
REF_DIAMETER = 0.015    # reference leak orifice for signatures (1.5 cm)
CD = 0.75               # orifice discharge coefficient
G = 9.81
SIM_HOURS = 48          # day 1 = tank warm-up (discarded), day 2 = kept
STEP_S = 900            # 15-min hydraulic/report step
KEEP = slice(96, 192)   # 15-min steps of day 2

# ---------------------------------------------------------------- data split
# time-based split: nothing from validation may be used to FIT anything
TRAIN_END = "2018-09-30 23:59:59"   # train = Jan-Sep, validation = Oct-Dec
STEPS_PER_HOUR = 12                 # data is sampled every 5 minutes

# event-aligned blocked cross-validation: 4 contiguous blocks whose edges lie in
# leak-free gaps, so every leak event lies entirely inside ONE block
FOLD_EDGES = ["2018-01-01", "2018-03-15", "2018-06-14", "2018-09-11", "2019-01-01"]
PURGE_HOURS = 24                    # training rows this close to a validation block are dropped
