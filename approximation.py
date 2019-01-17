import sys
import numpy as np
import math
from operator import itemgetter
from data import load_all_entries, ReutersEntry, ALLOWED_CHARS, SAVE_FOLDER
from ssk import normalized_ssk, ssk
from enum import Enum
import random
import time
from operator import itemgetter
import itertools
import pickle
import multiprocessing as mp

class FileEnum(Enum):
    AFFIX = ".s_doc_table.pkl"
    K_3_100_FIRST_ALL_S = "k_3_100_first_docs_all_substrings"
    DOC_DOC_SSK_K_3 = "doc_doc_ssk_k_3_l_0_5"
    SUBSTRINGS_INDEX = "pickels/all_sub_strings_index.pkl"


def contiguos_from_first_100_entries(all_entries):
    return contiguos_from_entries(all_entries[0:100])

def contiguos_from_entries(entries):
    all_strings = dict()
    all_3_grams = dict()
    for entry in entries:
        if entry.stride_3_grams is None:
            continue
        for gram, occurences in entry.stride_3_grams.items():
            if gram not in all_3_grams:
                all_3_grams[gram] = 0
            all_3_grams[gram] += occurences
    gram_tuples = [ (k,v) for k,v in all_3_grams.items() ]
    gram_tuples = sorted(gram_tuples, key=itemgetter(1), reverse=True)
    return [ s for (s,_) in gram_tuples ]

def gram_similarity(K1, K2):
    #assuming that S is a mxm matrix
    sum_k1_k1 = 0.0
    sum_k2_k2 = 0.0
    sum_k1_k2 = 0.0
    m = K1.shape[0]
    for i in range(m):
        # print("i: %d" % m)
        for j in range(m):
            sum_k1_k1 += K1[i][j] * K1[i][j]
            sum_k1_k2 += K1[i][j] * K2[i][j]
            sum_k2_k2 += K2[i][j] * K2[i][j]
    return sum_k1_k2 / math.sqrt(sum_k1_k1 * sum_k2_k2)

'''
Takes a set S and a list of documents
creates an approximation of a kernel using
K(x,z) = sum [ For all s in S K(x,s) * K(z,s) ]
'''
# def approximate_matrix(S, documents, k, l):
def approximate_matrix():
    # s_doc_table = compute_s_doc_table(S,documents,k,l)
    s_doc_table = load_s_doc_table()
    no_substrings, no_documents = s_doc_table.shape
    gram_matrix = np.zeros((no_documents, no_documents))
    for i,j in itertools.product(range(no_documents), range(no_documents)):
        # print("Calculating %d,%d" % (i,j))
        summa_s_t = 0.0
        summa_s_s = 0.0
        summa_t_t = 0.0
        for m in range(no_substrings):
            summa_s_t += s_doc_table[m,i] * s_doc_table[m,j]
            summa_s_s += 2 * s_doc_table[m,i]
            summa_t_t += 2 * s_doc_table[m,j]
        gram_matrix[i,j] = summa_s_t / ( np.sqrt(summa_s_s * summa_t_t))
    return gram_matrix


    # path = "%s/%s%s" % (SAVE_FOLDER, FileEnum.K_3_100_FIRST_ALL_S,FileEnum.AFFIX)
    # f = open(path, 'wb')
    # pickle.dump(s_doc_table, f)
    # f.close()


output = mp.Queue()
def __DxS_worker__(thread_id, S, k, l, DxD, arg):
    ''' arg[0]: doc_bucket  '''
    ''' arg[1]: K_ss        '''
    ''' arg[2]: start index '''
    ''' arg[3]: end index   '''

    print(f'Thread {thread_id} staring!')

    average_counter = 0.0
    average_total = 0.0
    average_running = 0.0

    offset = '\t\t' * thread_id

    doc_finished = 0

    sub_Ds_table = np.zeros((len(S), len(arg[0])))
    for i, doc in enumerate(arg[0]):
        time_start = time.time()
        for j, K_ss in enumerate(arg[1]):

            K_Ds = ssk(doc[1], S[j], k, l)
            sub_Ds_table[j, i] = K_Ds / np.sqrt(DxD[doc[0]] * K_ss)


            if j % 500 == 0:
                print(offset + f"worker {thread_id} at: {j}")

        doc_finished += 1
        print(offset + f'Worker {thread_id} finished doc {doc[0]} in {time.time()-time_start:.2f}')
        print(offset + f'Doc {doc_finished}/{len(arg[0])}')
    output.put((thread_id, sub_Ds_table))

def __get_worker_args__(docs, S, k, l, nworkers):

    # precalc sxs for all cores
    t0 = time.time()
    K_ss = np.zeros(len(S))
    for i in range(len(S)):
        K_ss[i] = ssk(S[i], S[i], k, l)
    print(f'K(s,s) calculation completed in {time.time() - t0}s')
    print(f'Splitting data for {nworkers} cores')

    n_docs = len(docs)
    doc_lengths = [len(d[1]) for d in docs]
    doc_total_length = sum(doc_lengths)
    split_threshold = sum(doc_lengths) / nworkers

    print('-' * 70)
    print('SPLIT summary:')

    args = []
    doc_bucket = []
    split_size = 0
    start = 0
    acc_size = 0
    for i, length in enumerate(doc_lengths):
        doc_bucket.append(docs[i])
        split_size += length
        acc_size += length
        if split_threshold < split_size or i == n_docs-1:
            # store args
            args.append((doc_bucket, K_ss, start, i))
            print(f'SPLIT: size: {split_size} docs included: {len(doc_bucket)}\tIndex from {start} to {i} ')
            split_size = 0
            doc_bucket = []
            start = i + 1
    print(f'Total size: {doc_total_length}, accumulated size: {acc_size}')
    print(f'Total docs: {n_docs} i.e last document index is {n_docs-1}')
    print('-' * 70)
    assert acc_size == doc_total_length
    return args

def precompute_DxS_table(documents, S, k, l, nworkers=3):

    #DxS_table = np.zeros((len(S), len(documents)))
    args = __get_worker_args__(documents, S, k, l, nworkers)
    DxD = load_precalc_DxD(k)


    processes = [mp.Process(target=__DxS_worker__, args=(i, S, k, l, DxD, arg)) for i, arg in enumerate(args)]
    for p in processes: p.start()
    for p in processes: p.join()
    results = sorted([output.get() for p in processes])

    Ds_table = np.zeros((len(S), len(documents)))
    for idx, arg in enumerate(args):
        print('table', Ds_table[:, arg[2]:arg[3]+1].shape)
        print('res  ', results[idx][1].shape)
        Ds_table[:, arg[2]:arg[3]+1] = results[idx][1]
    return SD_table



def create_all_substrings():
    l = len(ALLOWED_CHARS)
    all_subs = []
    for i in range(l):
        for j in range(l):
            for k in range(l):
                all_subs.append("%s%s%s" % (ALLOWED_CHARS[i],ALLOWED_CHARS[j],ALLOWED_CHARS[k]))
    return all_subs

def save_sub_string_index():
    all_substrings = create_all_substrings()
    index_mapping = dict()
    for i in range(len(all_substrings)):
        index_mapping[all_substrings[i]] = i
    f = open("pickels/all_sub_strings_index.pkl", 'wb')
    pickle.dump(index_mapping, f)
    f.close()

def load_sub_string_index():
    with open("pickels/all_sub_strings_index.pkl", 'rb') as f:
        mapping = pickle.load(f)
    return mapping


def precalc_s_s(documents,k,l):
    s_s = dict()
    counter = 1
    for (doc_id, doc) in documents:
        print("Working on document %d with id: %d" % (counter, doc_id))
        s_s[doc_id] = ssk(doc, doc, k,l)
        counter += 1
    f = open('pickels/s_s_k_3_l_0_5_all_documents.pkl', 'wb')
    pickle.dump(s_s, f)
    f.close()

def load_precalc_DxD(k):
    with open(f'pickels/DxD_k_{k}_l_0_5_all_documents.pkl', 'rb') as f:
        s_s = pickle.load(f)
    return s_s

def load_s_doc_table():
    f = open('pickels/s_doc_table_all_substrings_100_first_set_k_3_lambda_0_5.pkl', 'rb')
    s_doc = pickle.load(f)
    return s_doc

def load_pickle(filename):
    f = open(filename, 'rb')
    pick = pickle.load(f)
    f.close()
    return pick

def get_n_grams_in_100_first_docs():
    s_doc = load_s_doc_table()
    summed = np.sum(s_doc, axis=1)
    mapping = load_sub_string_index()
    inv_mapping = {v: k for k, v in mapping.items()}
    n_gram_count = 0
    n_gram_indexes = []
    for i in range(len(summed)):
        if summed[i] > 1e-2:
            n_gram_count += 1
            n_gram_indexes.append((inv_mapping[i], summed[i]))
    substrings = create_all_substrings()
    top_sorted_n_grams = sorted(n_gram_indexes, key=itemgetter(1), reverse=True)
    top_sorted_only_n_gram = [ k[0] for k in top_sorted_n_grams ]
    inv_sorted_n_grams = sorted(n_gram_indexes, key=itemgetter(1))
    inv_sorted_only_n_gram = [ k[0] for k in inv_sorted_n_grams ]

    # prefix_top_features = "pickels/gram_matrix_%d_top_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl"
    prefix_inv_features = "pickels/approx_for_graph/gram_matrix_%d_inv_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl"
    # prefix_rnd_features = "pickels/gram_matrix_%d_rnd_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl"
    grams = []
    feature_range = [ x for x in range(5, 200, 5) ]
    # feature_range_2 = [ x for x in range(200, 10200, 200) ]
    feature_range_2 = [ x for x in range(200, 3000, 200) ]
    feature_range_3 = [ x for x in range(3000, 7000, 1000) ]
    feature_range.extend(feature_range_2)
    feature_range.extend(feature_range_3)
    feature_range.append(10000)
    for features in feature_range:
        # print("Calculating for %d" % features)
        # top_ssk = approximate_matrix_from_subset(top_sorted_only_n_gram[:features])
        # top_ssk_name = prefix_top_features % features
        # f = open(top_ssk_name, 'wb')
        # pickle.dump(top_ssk,f)
        # f.close()
        inv_ssk = approximate_matrix_from_subset(inv_sorted_only_n_gram[:features])
        inv_ssk_name = prefix_inv_features % features
        f = open(inv_ssk_name, 'wb')
        pickle.dump(inv_ssk,f)
        f.close()
        # rnd_ssk = approximate_matrix_from_subset(random.sample(top_sorted_only_n_gram, features))
        # rnd_ssk_name = prefix_rnd_features % features
        # f = open(rnd_ssk_name, 'wb')
        # pickle.dump(rnd_ssk,f)
        # f.close()

    # gram_matrix = approximate_matrix_from_subset(grams)
    # f = open('pickels/stuff', 'wb')
    # pickle.dump(gram_matrix,f)
    # f.close()


def approximate_matrix_from_subset(subset):
    # s_doc_table = compute_s_doc_table(S,documents,k,l)
    s_doc_table = load_s_doc_table()
    _ , no_documents = s_doc_table.shape
    no_substrings = len(subset)
    gram_matrix = np.zeros((no_documents, no_documents))
    mapping = load_sub_string_index()
    for i,j in itertools.product(range(no_documents), range(no_documents)):
        # print("Calculating %d,%d" %
        summa_s_t = 0.0
        summa_s_s = 0.0
        summa_t_t = 0.0
        for ngram in subset:
            invert_m = mapping[ngram]
            summa_s_t += s_doc_table[invert_m,i] * s_doc_table[invert_m,j]
            summa_s_s += 2 * s_doc_table[invert_m,i]
            summa_t_t += 2 * s_doc_table[invert_m,j]
        gram_matrix[i,j] = summa_s_t / ( np.sqrt(summa_s_s * summa_t_t))
    return gram_matrix


if __name__ == '__main__':
    #get_n_grams_in_100_first_docs()
    #Comparing gram matrices
    # full_kernel = load_pickle("pickels/100_doc_gram_matrix")
    # approx_kernel_all_features = load_pickle("pickels/gram_matrix_all_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl")
    # approx_kernel_11005_features = load_pickle("pickels/gram_matrix_11005_top_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl")
    # gram_si = gram_similarity(full_kernel, full_kernel)
    # gram_sim = gram_similarity(full_kernel, approx_kernel_all_features)
    # gram_sim2 = gram_similarity(full_kernel, approx_kernel_11005_features)
    # print("Simmilarity full kernel vs full kernel: %f" % gram_si)
    # print("Simmilarity full kernel vs all features: %f" % gram_sim)
    # print("Simmilarity full kernel vs 11005 features: %f" % gram_sim2)
    # entries = load_all_entries()
    # train_entries = [ (x.id, x.clean_body) for x in entries if x.lewis_split == 'TRAIN' ]
    # substrings = create_all_substrings()

    # gram_matrix = approximate_matrix()
    # f = open('pickels/gram_matrix_all_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl', 'wb')
    # pickle.dump(gram_matrix,f)
    # f.close()

    #gram_matrix = approximate_matrix()
    #f = open('pickels/gram_matrix_all_features_100_first_documents_k_3_lambda_0_5.pkl', 'wb')
    #pickle.dump(gram_matrix,f)
    #f.close()

    entries = load_all_entries()
    all_bodies = [(x.id, x.clean_body) for x in entries]
    #train_entries = [ (x.id, x.clean_body) for x in entries if x.lewis_split == 'TRAIN' ]
    substrings = create_all_substrings()

    SD_table = precompute_DxS_table(all_bodies, substrings, 3, 0.5, nworkers=7)
    f = open('pickels/s_D_table_full_k_3_lambda_0_5.pkl', 'wb')
    pickle.dump(SD_table, f)
    f.close()
    # precalc_s_s(all_bodies, 3, 0.5)
