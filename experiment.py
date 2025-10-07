import os
import numpy as np
import pickle
import time
import attacks
from matplotlib import pyplot as plt
import utils
from defense import generate_observations
from collections import Counter, defaultdict
from config import PRO_DATASET_FOLDER

HIGH_CORR = False


def load_pro_dataset(dataset_name):
    full_path = os.path.join(PRO_DATASET_FOLDER, dataset_name + '.pkl')
    if not os.path.exists(full_path):
        raise ValueError("The file {} does not exist".format(full_path))

    with open(full_path, "rb") as f:
        dataset, keywords, aux = pickle.load(f)

    return dataset, keywords, aux


def generate_keyword_queries(mode_query, frequencies, nqr, nkw):
    # nkw = int(frequencies.shape[0]) # frequencies.shape[0]/2
    # if mode_query == 'iid':
    #     assert frequencies.ndim == 1
    #     # queries = list(np.random.choice(list(range(nkw)), nqr, p=frequencies))

    #     queries = []
    #     while len(queries) < nqr:
    #         kwQuery = np.random.choice(list(range(nkw)), p=frequencies[:nkw]*2)
    #         docQuery = np.random.choice(kwToDocIds[kwQuery]) # for a skewed distribution, include that as p here
    #         queries.append(kwQuery)
    #         queries.append(docQuery)
        
    #     # Compress space of accessed documents.
    #     # e.g. let number of non-sampled documents = 30k, number of sampled = 3k. The sampled 3k could be any of the 30k
    #     # but we want to compress them into indices NUM_KEYWORDS:NUM_KEYWORDS+3000 so they can be used to index into freqMatrix
    #     docMap = {}
    #     currentIdx = nkw
    #     for i in range(nqr):
    #         if i%2==0: continue # skip keyword accesses
    #         if queries[i] not in docMap: 
    #             docMap[queries[i]] = currentIdx
    #             currentIdx += 1
    #         queries[i] = docMap[queries[i]]
        
    #     # print("queries[0:100] line43 experiment.py:", queries[0:100])
    #     # print("unique:", len(set(queries)))

    # mode_query should always be 'markov' for the scenarios we consider.
    if mode_query == 'markov':
        assert frequencies.ndim == 2

        ## TRANSCRIPT GENERATION
        queries = []

        while len(queries) < nqr:
            # pick a keyword
            kwQuery = np.random.choice(list(range(nkw)), p=frequencies[:nkw, nkw])
            # print(kwQuery, frequencies[0, nkw:])
            # print(len(list(range(nkw, len(frequencies)))), len(frequencies[kwQuery, nkw:]))

            # pick a document based on the keyword
            if HIGH_CORR: 
                # always take the *first* document containing the keyword
                docQuery = np.argwhere(frequencies[nkw:, kwQuery])[0][0]+nkw
            else: 
                # pick uniformly among the documents containing the keyword
                docQuery = np.random.choice(list(range(nkw, len(frequencies))), p=frequencies[nkw:, kwQuery]) # for a skewed distribution, include that as p here
            queries.append(kwQuery)
            queries.append(docQuery)

    # elif mode_query == 'each':
    #     queries = list(np.random.permutation(nkw))[:min(nqr, nkw)]
    else:
        raise ValueError("Frequencies has {:d} dimensions, only 1 or 2 allowed".format(frequencies.ndim))
    return queries

def build_frequencies_from_file(chosen_kw_indices, chosen_doc_indices, dataset, trends):
    num_keys = len(chosen_kw_indices) + len(chosen_doc_indices)
    freq_real = np.zeros((num_keys, num_keys))
    
    # filter trends to chosen kws, collapse 52 weeks of trend data to a 1d array
    trend_matrix = trends[chosen_kw_indices, :]
    for i_col in range(trend_matrix.shape[1]):
        if sum(trend_matrix[:, i_col]) == 0:
            print("The {:d}th column of the trend matrix adds up to zero, making it uniform!".format(i_col))
            trend_matrix[:, i_col] = 1 / len(chosen_kw_indices)
        else:
            trend_matrix[:, i_col] = trend_matrix[:, i_col] / sum(trend_matrix[:, i_col])
    kw_freq = np.mean(trend_matrix, axis=1)

    # build transitions from docs to kws. No matter which doc, probability vector of transiting to any kw is kw_freq
    for doc_idx in range(len(chosen_kw_indices), num_keys):
        freq_real[0:len(chosen_kw_indices), doc_idx] = kw_freq

    # build transitions from kws to docs
    if HIGH_CORR:
        # transition probability from kw to first doc containing it is 1, else 0
        for kw_idx, kw in enumerate(chosen_kw_indices):
            doc_selected_for_kw = False
            for doc_i, doc_n in enumerate(chosen_doc_indices):
                if doc_selected_for_kw: continue

                doc = dataset[doc_n]
                doc_idx = doc_i + len(chosen_kw_indices)
                if kw in doc:
                    freq_real[doc_idx, kw_idx] = 1
                    doc_selected_for_kw = True
    else:
        # transition probability from kw for n docs containing it is 1/n each
        for kw_idx, kw in enumerate(chosen_kw_indices):
            n = 0
            for doc_i, doc_n in enumerate(chosen_doc_indices):
                doc = dataset[doc_n]
                doc_idx = doc_i + len(chosen_kw_indices)
                if kw in doc:
                    n += 1
                    freq_real[doc_idx, kw_idx] = 1
            if (n > 0): # sum should be 1
                for doc_i, doc in enumerate(chosen_doc_indices):
                    doc_idx = doc_i + len(chosen_kw_indices)
                    freq_real[doc_idx, kw_idx] /= n

    # sanity check
    # column i = probability vector of transitioning from token i to other tokens (IHOP appendix D)
    # sum of column i should = 1 
    import math
    for r in range(num_keys):
        assert(math.isclose(sum(freq_real[:, r]), 1))   
    
    return freq_real, freq_real, freq_real

def old_build_frequencies_from_file(dataset_name, chosen_kw_indices, keywords, aux_dataset_info, mode_fs):
    def _process_markov_matrix(months):

        m = np.zeros((nkw, nkw))
        msink = np.zeros(nkw)
        for month in months:
            m += aux_dataset_info['transitions'][month][np.ix_(chosen_kw_indices, chosen_kw_indices)]
            msink += aux_dataset_info['transitions'][month][chosen_kw_indices, -1]

        sink_profile = msink / np.sum(msink)
        cols_sunk = [val[0] for val in np.argwhere(m.sum(axis=0) == 0)]
        m[:, cols_sunk] = np.ones(len(cols_sunk)) * msink.reshape(nkw, 1)

        # Add a certain probability of restart
        m = m / m.sum(axis=0)
        p_restart = 0.05  # DONE!
        m_new = (1 - p_restart) * m + p_restart * sink_profile.reshape(len(sink_profile), 1)

        # m_markov = m_new / m_new.sum(axis=0)
        ss = utils.get_steady_state(m_new)
        if any(ss < -1e-8):
            print(ss[ss < 0])
        print(m_new)
        return m_new

    nkw = len(chosen_kw_indices)
    if dataset_name in ('enron-full', 'lucene', 'bow-nytimes', 'articles1', 'movie-plots'):
        trend_matrix = aux_dataset_info['trends'][chosen_kw_indices, :]
        for i_col in range(trend_matrix.shape[1]):
            if sum(trend_matrix[:, i_col]) == 0:
                print("The {:d}th column of the trend matrix adds up to zero, making it uniform!".format(i_col))
                trend_matrix[:, i_col] = 1 / nkw
            else:
                trend_matrix[:, i_col] = trend_matrix[:, i_col] / sum(trend_matrix[:, i_col])
        if mode_fs == 'same':  # Take last year of data
            freq = np.mean(trend_matrix, axis=1)
            freq /= 2 # only half of the queries are keywords
            freq_adv = freq_cli = freq_real = np.append(freq, [0.5/nkw for i in range(nkw)])
            # print("freq_real exp:202:", freq_real)
        elif mode_fs == 'past':  # First half of year for adv, last half is real and client's
            freq_adv = np.mean(trend_matrix[:, -52:-26], axis=1)
            freq_cli = freq_real = np.mean(trend_matrix[:, -26:], axis=1)
        else:
            raise ValueError("Frequencies split mode '{:s}' not allowed for {:s}".format(mode_fs, dataset_name))
    elif dataset_name.startswith('wiki'):
        # category = dataset_name[5:]
        if mode_fs == 'same':
            months_real = range(7, 13)
            months_adv = range(7, 13)
        elif mode_fs == 'past':
            months_real = range(7, 13)
            months_adv = range(1, 7)
        elif mode_fs == 'same1':  # December vs December
            months_real = [12]
            months_adv = [12]
        elif mode_fs == 'past1':  # June vs December
            months_real = [12]
            months_adv = [6]
        else:
            raise ValueError("Frequencies split mode '{:s}' not allowed for {:s}".format(mode_fs, dataset_name))
        freq_adv = _process_markov_matrix(months_adv)
        freq_real = _process_markov_matrix(months_real)
        freq_cli = _process_markov_matrix(months_real)
    else:
        raise ValueError("No frequencies for dataset {:s}".format(dataset_name))
    return freq_adv, freq_cli, freq_real

def generate_train_test_data(gen_params):
    # nkw is the number of keywords in the datastore
    # ndocs is the nubmer of documents in the datastore
    # likely nkw ~= ndocs
    nkw = gen_params['nkw']
    ndoc = gen_params['ndoc']
    dataset_name = gen_params['dataset']
    mode_kw = gen_params['mode_kw']
    mode_ds = gen_params['mode_ds']
    freq_name = gen_params['freq']
    mode_fs = gen_params['mode_fs']

    # Load the dataset for this experiment
    dataset, keywords, aux_dataset_info = load_pro_dataset(dataset_name)

    # use top strategy to pick keywords - picking most popular minimizes likelihood that some kw appears in no doc
    chosen_kw_indices = list(range(nkw))
    # rand strategy: chosen_kw_indices = np.random.permutation(len(keywords))

    # use rand strategy to pick docs - could also do with top
    permutation = np.random.permutation(len(dataset))
    chosen_doc_indices = list(permutation[:ndoc])

    # get Markov transition matrix
    freq_adv, freq_cli, freq_real = build_frequencies_from_file(chosen_kw_indices, chosen_doc_indices, dataset, aux_dataset_info['trends'])

    full_data_adv = {'dataset': dataset,
                     'keywords': range(nkw+ndoc),
                     'frequencies': freq_adv,
                     'mode_query': gen_params['mode_query']}
    full_data_client = {'dataset': dataset,
                        'keywords': range(nkw+ndoc),
                        'frequencies': freq_cli}
    return full_data_adv, full_data_client, freq_real

def old_generate_train_test_data(gen_params):
    nkw = gen_params['nkw']
    dataset_name = gen_params['dataset']
    mode_kw = gen_params['mode_kw']
    mode_ds = gen_params['mode_ds']
    freq_name = gen_params['freq']
    mode_fs = gen_params['mode_fs']

    # Load the dataset for this experiment
    dataset, keywords, aux_dataset_info = load_pro_dataset(dataset_name)
    ndoc = len(dataset) if gen_params['ndoc'] == 'full' else min(len(dataset), gen_params['ndoc'])

    # Select the keys [previously: keywords] for this experiment
    if mode_kw == 'top':
        # Select nkw/2 keywords
        kw_counter = Counter([kw for document in dataset for kw in document])
        chosen_kw_indices = sorted(kw_counter.keys(), key=lambda x: kw_counter[x], reverse=True)[:int(nkw/2)] # in 'top' mode, this is the same as range(nkw/2)
        # print("chosen_kw_indices:", chosen_kw_indices)

        # Select nkw/2 documents: downsample document set to reduce keyword universe size
        chosen_doc_indices = np.random.choice(range(len(dataset)), size=int(nkw/2), replace=False)
        chosen_doc_indices.sort()
        print("nkw/2:", int(nkw/2))
        # print("chosen_doc_indices:", chosen_doc_indices)

        # Get kwId -> docIds containing that kw mapping. Select only documents in chosen_doc_indices
        kwIdToDocIds = defaultdict(list)
        # kw_to_kw_id = {kw: kw_id for kw_id, kw in enumerate(keywords)}
        for doc_id, doc_kws in enumerate(dataset):
            if (doc_id not in chosen_doc_indices): continue
            for kw in set(doc_kws):
                kwIdToDocIds[kw].append(doc_id)
        print("kwIdToDocIds[0] line149:", kwIdToDocIds[0])

    elif mode_kw == 'rand':
        permutation = np.random.permutation(len(keywords))
        chosen_kw_indices = list(permutation[:nkw])
    else:
        raise ValueError("Keyword selection mode '{:s}' not allowed".format(mode_kw))

    # Get client dataset and adversary's auxiliary dataset
    dataset = [dataset[i] for i in np.random.permutation(len(dataset))[:ndoc]]
    if mode_ds.startswith('same'):
        percentage = 100 if mode_ds == 'same' else int(mode_ds[4:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[:int(len(dataset) * percentage / 100)]]
        data_adv = dataset_selection
        data_cli = dataset_selection
    elif mode_ds.startswith('common'):
        percentage = 50 if mode_ds == 'common' else int(mode_ds[6:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[:int(len(dataset) * percentage / 100)]]
        data_adv = dataset_selection
        data_cli = dataset
    elif mode_ds.startswith('split'):
        if mode_ds.startswith('splitn'):
            ndocs_adv = int(mode_ds[6:])
        else:
            percentage = 50 if mode_ds == 'split' else int(mode_ds[5:])
            assert 0 < percentage < 100
            ndocs_adv = int(len(dataset) * percentage / 100)
        permutation = np.random.permutation(len(dataset))
        data_adv = [dataset[i] for i in permutation[:ndocs_adv]]
        data_cli = [dataset[i] for i in permutation[ndocs_adv:]]
    else:
        raise ValueError("Dataset split mode '{:s}' not allowed".format(mode_ds))

    # Load query frequency info
    if freq_name == 'file':
        freq_adv, freq_cli, freq_real = build_frequencies_from_file(dataset_name, chosen_kw_indices, keywords, aux_dataset_info, mode_fs)
    elif freq_name.startswith('zipf'):
        shift = int(freq_name[5:]) if freq_name.startswith('zipfs') else 0  # zipfs200 is a zipf with 200 shift
        aux = np.array([1 / (i + shift + 1) for i in range(nkw)])
        freq_adv = freq_cli = freq_real = aux / np.sum(aux)
    elif freq_name == 'none':
        freq_adv, freq_cli, freq_real = None, None, np.ones(nkw) / nkw
    else:
        raise ValueError("Frequency name '{:s}' not implemented yet".format(freq_name))

    full_data_adv = {'dataset': data_adv,
                     'keywords': range(nkw), #chosen_kw_indices,
                     'frequencies': freq_adv,
                     'mode_query': gen_params['mode_query']}
    full_data_client = {'dataset': data_cli,
                        'keywords': range(nkw), # chosen_kw_indices,
                        'frequencies': freq_cli}
    return full_data_adv, full_data_client, freq_real # kwIdToDocIds


def run_attack(attack_name, **kwargs):
    if attack_name == 'freq':
        return attacks.freq_attack(**kwargs)
    elif attack_name == 'sap':
        return attacks.sap_attack(**kwargs)
    elif attack_name == 'ihop':
        return attacks.ihop_attack(**kwargs)
    elif attack_name == 'umemaya':
        return attacks.umemaya_attack(**kwargs)
    elif attack_name == 'fastpfp':
        return attacks.fastfpf_attack(**kwargs)
    elif attack_name == 'ikk':
        return attacks.ikk_attack(**kwargs)
    elif attack_name == 'graphm':
        return attacks.graphm_attack(**kwargs)
    else:
        raise ValueError("Attack name '{:s}' not recognized".format(attack_name))


def run_experiment(exp_param, seed, debug_mode=False):
    v_print = print if debug_mode else lambda *a, **k: None

    t0 = time.time()
    np.random.seed(seed)
    full_data_adv, full_data_client, freq_real = generate_train_test_data(exp_param.gen_params)
    v_print("Generated train-test data: adv dataset {:d}, client dataset {:d} ({:.1f} secs)".format(len(full_data_adv['dataset']),
                                                                                                    len(full_data_client['dataset']),
                                                                                                    time.time() - t0))

    real_queries = generate_keyword_queries(exp_param.gen_params['mode_query'], freq_real, exp_param.gen_params['nqr'], exp_param.gen_params['nkw'])
    v_print("Generated {:d} real queries ({:.1f} secs)".format(len(real_queries), time.time() - t0))

    observations, bw_overhead, real_and_dummy_queries = generate_observations(full_data_client, exp_param.def_params, real_queries)
    v_print("Applied defense ({:.1f} secs)".format(time.time() - t0))

    keyword_predictions_for_each_query = run_attack(exp_param.att_params['name'], obs=observations, aux=full_data_adv, exp_params=exp_param)
    v_print("Done running attack ({:.1f} secs)".format(time.time() - t0))
    time_exp = time.time() - t0

    # Compute accuracy
    if type(keyword_predictions_for_each_query) == list and type(keyword_predictions_for_each_query[0]) != list:
        acc_vector = np.array([1 if query == prediction else 0 for query, prediction in zip(real_and_dummy_queries, keyword_predictions_for_each_query)])
        acc_un_vector = np.array([np.mean(acc_vector[real_and_dummy_queries == i]) for i in set(real_and_dummy_queries)])
        accuracy = np.mean(acc_vector)
        accuracy_un = np.mean(acc_un_vector)
        return accuracy, accuracy_un, time_exp
    elif type(keyword_predictions_for_each_query) == list and type(keyword_predictions_for_each_query[0]) == list:
        acc_list, acc_un_list = [], []
        for pred in keyword_predictions_for_each_query:
            acc_vector = np.array([1 if query == prediction else 0 for query, prediction in zip(real_and_dummy_queries, pred)])
            # print(np.mean(np.array([1 if query == prediction else 0 for query, prediction in zip(real_and_dummy_queries, pred)])), np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(real_and_dummy_queries, pred)])))
            acc_un_vector = np.array([np.mean(acc_vector[real_and_dummy_queries == i]) for i in set(real_and_dummy_queries)])
            acc_list.append(np.mean(acc_vector))
            acc_un_list.append(np.mean(acc_un_vector))
        return acc_list, acc_un_list, time_exp
    else:
        return -1, -1, -1
