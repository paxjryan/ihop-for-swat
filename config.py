RAW_DATASET_FOLDER = 'datasets_raw'
PRE_DATASET_FOLDER = 'datasets_pre'
PRO_DATASET_FOLDER = 'datasets_pro'

# debug.py
DEFENSE = 'pancake'
NKW = 250
NQR = 5_000_000
NITERS = 1_000
NITER_LIST = [0, 10, 100, 500, 1000]
PFREE = 0.25

EXPERIMENT_NAME = 'PancakeLowCorr5M'
EXPERIMENT_FOLDER = 'out/' + EXPERIMENT_NAME + '/'
NRUNS = 2 # 10

# experiment.py
CORR_LEVEL = 'low'  # 'high': each kw only transitions to one (first/random) doc containing it - see HIGH_CORR_PERMUTE
                    # 'mid' : each kw can transition to any doc containing it, but weighted exponentially
                    # 'low' : each kw can transition to any doc containing it (weighted equally)
HIGH_CORR_PERMUTE = False # If CORR_LEVEL = 'high' and HIGH_CORR_PERMUTE = True, each kw only transitions to random doc containing it
                          # If False, each kw only transitions to first doc containing it

DISPLAY_ACC_VECTORS = False
SAVE_ACC_VECTORS = True

# process_aux.py
DISPLAY_AUX_GRAPH = False
SAVE_AUX_GRAPH = True

# process_obs.py
DISPLAY_OBS_GRAPH = False
SAVE_OBS_GRAPH = True
MOD_FOBS = False
