RAW_DATASET_FOLDER = 'datasets_raw'
PRE_DATASET_FOLDER = 'datasets_pre'
PRO_DATASET_FOLDER = 'datasets_pro'

# debug.py
DEFENSE = 'pancake'
NKW = 250
NQR = 100_000
NITERS = 100
NITER_LIST = [0, 100] #, 500, 1000] #, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
PFREE = 0.25

USE_THETA_DECORR = True
THETA = 0

EXPERIMENT_NAME = 'test'
EXPERIMENT_FOLDER = 'out/' + EXPERIMENT_NAME + '/'
NRUNS = 1 # 10

# experiment.py
CORR_LEVEL = 'low'  # 'high': each kw only transitions to one (first/random) doc containing it - see HIGH_CORR_PERMUTE
                    # 'mid' : each kw can transition to any doc containing it, but weighted exponentially
                    # 'low' : each kw can transition to any doc containing it (weighted equally)
HIGH_CORR_PERMUTE = False # If CORR_LEVEL = 'high' and HIGH_CORR_PERMUTE = True, each kw only transitions to random doc containing it
                          # If False, each kw only transitions to first doc containing it

DISPLAY_ACC_VECTORS = False
SAVE_ACC_VECTORS = False

BASE_SEED = 52 # debug.py runs deterministically; introduce randomness in case a run gets interrupted and we want to restart without redoing all of the previously-done randomness

# process_aux.py
DISPLAY_AUX_GRAPH = False
SAVE_AUX_GRAPH = True

# process_obs.py
DISPLAY_OBS_GRAPH = False
SAVE_OBS_GRAPH = True
MOD_FOBS = False
