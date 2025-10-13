import numpy as np
from collections import Counter
import scipy.stats
from sys import float_info
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt #added

def traces_to_binary(traces_flattened, n_docs_test):
    # TODO: do this more efficiently
    binary_traces = np.zeros((len(traces_flattened), n_docs_test))
    for i_trace, trace in enumerate(traces_flattened):
        for doc_id in trace:
            binary_traces[i_trace, doc_id] = 1
    return binary_traces


def compute_Vobs(trace_type, token_info, n_docs_test):
    ntok = len(token_info)
    if trace_type == 'ap_unique':
        database_matrix = np.zeros((n_docs_test, ntok))
        for tag in token_info:
            for doc_id in token_info[tag]:
                database_matrix[doc_id, tag] = 1
        Vobs = np.matmul(database_matrix.T, database_matrix) / n_docs_test
    elif trace_type == 'tok_vol':
        Vobs = np.zeros((ntok, ntok))
        for j in range(ntok):
            Vobs[j, j] = token_info[j] / n_docs_test
    elif trace_type == 'ap_osse':
        Vobs = np.zeros((ntok, ntok))
        for i in range(ntok):
            for j in range(ntok):
                if i > j:
                    Vobs[i, j] = Vobs[j, i]
                elif i == j:
                    Vobs[i, i] = np.mean([len(trace) for trace in token_info[i]]) / n_docs_test
                else:
                    Vobs[i, j] = np.mean([len(set(trace1) & set(trace2)) for trace1 in token_info[i] for trace2 in token_info[j]]) / n_docs_test
    else:
        raise ValueError("trace_type={:s} not recognized".format(trace_type))
    return Vobs


def compute_vobs(trace_type, token_info, n_docs_test):
    if trace_type == 'ap_unique':
        vobs = [len(token_info[tok]) / n_docs_test for tok in token_info]
    elif trace_type == 'tok_vol':
        vobs = [token_info[tok] / n_docs_test for tok in token_info]
    elif trace_type == 'ap_osse':
        vobs = [np.mean([len(doc_set) for doc_set in token_info[tag]]) / n_docs_test for tag in token_info]
    else:
        raise ValueError("trace_type={:s} not recognized".format(trace_type))
    return vobs


def compute_fobs(def_name, token_trace, n_tokens):
    counter = Counter(token_trace)
    fobs = np.array([counter[j] / len(token_trace) for j in range(n_tokens)])
    return fobs


def compute_Fobs(def_name, token_trace, n_tokens):
    Fobs = np.zeros((n_tokens, n_tokens))
    nq_per_tok = np.zeros(n_tokens)
    counter = Counter(token_trace[:-1])  # We do not take the last one, because we do not know the transition from that one

    # print("token_trace[0:100]:", token_trace[0:100])

    if def_name != 'pancake':
        mj_test = np.histogram2d(token_trace[1:], token_trace[:-1], bins=(range(n_tokens + 1), range(n_tokens + 1)))[0] / (len(token_trace) - 1)
    else:
        mj_test = np.zeros((n_tokens, n_tokens))
        for i in range(3):
            for j in range(3):
                mj_test += np.histogram2d(token_trace[3 + i::3], token_trace[j:-3:3], bins=(range(n_tokens + 1), range(n_tokens + 1)))[0]
    
    # values are hardcoded for now while I determine how to get total number of kw/doc/dummy replicas
    # 0:kw_replicas are keyword replicas. kw:replicas:real_replicas are document replicas. real_replicas: are dummy replicas
    kw_replicas = 397
    real_replicas = 792
    doc_replicas = real_replicas-kw_replicas

    # Set transition frequencies from kw->kw and doc->doc replicas in Fobs to 0
    # mj_test[0:kw_replicas, 0:kw_replicas] = np.zeros((kw_replicas, kw_replicas))
    # mj_test[kw_replicas:real_replicas, kw_replicas:real_replicas] = np.zeros((doc_replicas,doc_replicas))
    
    # Set transition frequencies from dummy->any and any->dummy replicas in Fobs to 0
    # (not convinced about this)
    # mj_test[real_replicas:, :] = np.zeros((1000-real_replicas, 1000))
    # mj_test[:, real_replicas:] = np.zeros((1000, 1000-real_replicas))

    for j in range(n_tokens):
        nq_per_tok[j] = np.sum(mj_test[:, j])
        if np.sum(mj_test[:, j]) > 0:
            Fobs[:, j] = mj_test[:, j] / np.sum(mj_test[:, j])

    # Display heatmap of Fobs
    plt.imshow(Fobs, cmap='Greys_r', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap with Matplotlib")
    plt.show()

    return nq_per_tok, Fobs


def process_traces(obs, aux, def_params):
    def _process_traces_with_search_pattern_leakage_given_access_pattern(traces, aux_dict):

        """tag_info is a dict [tag] -> AP (list of doc ids)"""
        token_trace = []
        seen_id_to_token_id = {}
        token_info = {}
        token_id = 0
        ids = [x[0] for x in traces]
        print(set(ids))
        print(len(set(ids)))
        for id, ap in traces:
            ap_sorted = tuple(sorted(ap))
            if id not in seen_id_to_token_id:
                seen_id_to_token_id[id] = token_id
                token_info[token_id] = ap_sorted
                token_id += 1
            token_trace.append(seen_id_to_token_id[id])
            # Don't rename seen tokens, so when we generate Fobs we know which replicas are kws/docs/dummies
            # token_trace.append(id)
        return token_trace, token_info

    def _process_traces_with_search_pattern_leakage_given_volume(traces):
        """tag_info is a dict [tag] -> response volume"""

        token_traces = []
        seen_ids_to_token_id = {}
        token_info = {}
        token_id = 0
        # print("process_obs, line 110, delete that line!")
        for id, vol in traces:
            if id not in seen_ids_to_token_id:
                # token_id = id
                seen_ids_to_token_id[id] = token_id
                token_info[token_id] = vol
                token_id += 1
            # token_traces.append(seen_ids_to_token_id[id])
            # Don't rename seen tokens, so when we generate Fobs we know which replicas are kws/docs/dummies
            token_traces.append(id)
        return token_traces, token_info
    
        # token_traces = []
        # seen_ids_to_token_id = {}
        # token_info = {}
        # kw_token_id = 0
        # doc_token_id = 0
        # # print("process_obs, line 110, delete that line!")
        # print("len(traces), should be num queries:", len(traces))
        # for id, vol in traces:
        #     if id not in seen_ids_to_token_id:
        #         # token_id = id
        #         token_id = kw_token_id if id < len(traces)/6 else doc_token_id 
        #         seen_ids_to_token_id[id] = token_id
        #         token_info[token_id] = vol
        #         if id < len(traces)/6: kw_token_id += 1
        #         else: doc_token_id += 1
        #     token_traces.append(seen_ids_to_token_id[id])
        # print(seen_ids_to_token_id)
        # return token_traces, token_info

    def _process_traces_by_clustering_given_access_pattern(traces, nclusters, ndocs_obs):
        """tag_info is a dict [tag -> cluster center]"""

        binary_traces = traces_to_binary(traces, ndocs_obs)
        kmeans = KMeans(n_clusters=nclusters).fit(binary_traces)
        labels = kmeans.labels_

        # self.tag_cluster_sizes = np.histogram(list(labels), bins=range(nclusters + 1))[0]

        token_traces = list(labels)
        token_info = {i: [trace for j, trace in enumerate(traces) if i == labels[j]] for i in range(nclusters)}

        return token_traces, token_info

    if obs['trace_type'] == 'ap_unique':
        token_trace, token_info = _process_traces_with_search_pattern_leakage_given_access_pattern(obs['traces'], None)
    elif obs['trace_type'] == 'ap_osse':
        token_trace, token_info = _process_traces_by_clustering_given_access_pattern(obs['traces'], obs['n_distinct'], obs['ndocs'])
    elif obs['trace_type'] == 'tok_vol':
        token_trace, token_info = _process_traces_with_search_pattern_leakage_given_volume(obs['traces'])
    else:
        raise ValueError("trace_type={:s} not recognized".format(str(obs['trace_type'])))

    return token_trace, token_info
