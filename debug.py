import os
import time
from experiment import run_experiment
import numpy as np
from exp_params import ExpParams
from config import *

def print_exp_to_run(parameter_dict, n_runs):
    for key in parameter_dict:
        print('  {:s}: {}'.format(key, parameter_dict[key]))
    print("* Number of runs: {:d}".format(n_runs))


if __name__ == "__main__":

    os.system('mesg n')

    time_init = time.time()
    PRO_DATASETS_PATH = 'datasets_pro'
    EXPERIMENTS_PATH = 'results'

    exp_params = ExpParams()
    ###### Simple Enron experiment; accuracy for seed=0 should be 0.058, 0.174, 0.904, and 0.952 for niters=0, 10, 100, and 1000, respectively.
    # exp_params.set_general_params(dataset='enron-full', nkw=500, nqr=500, ndoc=30_000, freq='file', mode_ds='splitn10000',
    #                               mode_fs='past', mode_kw='rand', mode_query='iid')
    # exp_params.set_defense_params('none')
    # attack_list = [('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25})]
    # niter_list = [0, 10, 100, 1000]

    ##### Other examples of param initialization
    # exp_params.set_general_params(dataset='bow-nytimes', nkw=500, nqr=500, freq='file', mode_ds='split50', mode_fs='past', mode_kw='rand', mode_query='each')
    # exp_params.set_general_params(dataset='wiki_sec', nkw=500, nqr=100_000, freq='file', mode_ds='same', mode_fs='same', mode_kw='rand', mode_query='markov')
    # exp_params.set_general_params(dataset='enron-full', nkw=200, nqr=200, ndoc=30_000, freq='file', mode_ds='splitn10000', mode_fs='past', mode_kw='rand', mode_query='iid')
    # exp_params.set_general_params(dataset='enron-full', nkw=1000, nqr=1000, ndoc=30_000, freq='none', mode_ds='splitn10000', mode_fs='past', mode_kw='rand', mode_query='each')

    # Added
    exp_params.set_defense_params(DEFENSE)
    exp_params.set_general_params(dataset='enron-full', nkw=NKW, ndoc=NKW, nqr=NQR, freq='file', mode_ds='same', mode_fs='same', mode_kw='rand', mode_query='markov') # IHOP also does nqr=500
    attack_list = [('ihop', {'mode': 'Freq', 'niters': NITERS, 'pfree': PFREE})] # may want to go up to 10k iters? - see IHOP fig. 9
    niter_list = NITER_LIST

    ##### DEFENSE EXAMPLES
    # exp_params.set_defense_params('none')
    # exp_params.set_defense_params('pancake')
    # exp_params.set_defense_params('clrz', tpr=0.9999, fpr=0.02)

    ###### ATTACK LIST EXAMPLE
    # attack_list = [
    #     ('freq', {}),
    #     ('sap', {'alpha': 0.}),
    #     ('sap', {'alpha': 0.5}),
    #     ('sap', {'alpha': 1}),
    #     ('ihop', {'mode': 'Vol_freq', 'niters': 1000, 'pfree': 0.25}),
    #     ('ihop', {'mode': 'Vol', 'niters': 100, 'pfree': 0.25}),
    #     ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
    #     ('umemaya', {}),
    #     ('fastpfp', {}),
    #     ('ikk', {'cooling': 0.99}),
    #     ('graphm', {'alpha': 0.5}),
    # ]

    os.makedirs(os.path.dirname(EXPERIMENT_FOLDER), exist_ok=True)

    # save config for this experiment
    with open(EXPERIMENT_FOLDER + '/config.txt', "w") as configOutFile:
        with open("config.py", "r") as conf:
            print(conf.read(), file=configOutFile)

    # save results for this experiment
    with open(EXPERIMENT_FOLDER + '/output.txt', "a") as f:
        np.set_printoptions(precision=4)
        print(exp_params, file=f)

        acc_list = [[] for _ in attack_list]
        for seed in range(NRUNS):
            print("seed:", seed)
            print("Seed: ", seed, file=f)

            for i_att, (att, att_p) in enumerate(attack_list):
                exp_params.set_attack_params(att, **att_p)
                exp_params.att_params['niter_list'] = niter_list
                acc, accu, time_exp = run_experiment(exp_params, seed=seed, debug_mode=True)
                if type(acc) == list:
                    acc_list[i_att].append((acc[-1], accu[-1]))
                    for acc, accu, niters in zip(acc, accu, exp_params.att_params['niter_list']):
                        print("{:d}-{:d}) {:s}, acc={:.3f}, accu={:.3f} ({:.2f} secs)".format(seed, niters, att, acc, accu, time_exp), file=f)
                else:
                    acc_list[i_att].append((acc, accu))
                    print("{:d}) {:s}, acc={:.3f}, accu={:.3f} ({:.2f} secs)".format(seed, att, acc, accu, time_exp), file=f)

        print("Summary of results:", file=f)
        for i_att, (att, att_p) in enumerate(attack_list):
            print("{:s}: avg acc={:.3f}, avg accu={:.3f}".format(att, *[np.mean(aux) for aux in zip(*acc_list[i_att])]), file=f)
