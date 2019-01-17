import sys
import numpy as np
import math
from operator import itemgetter
from data import load_all_entries, ReutersEntry, ALLOWED_CHARS, SAVE_FOLDER
from ssk import normalized_ssk, ssk
from enum import Enum
import time
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
    sum_k1_k1 = 0
    sum_k2_k2 = 0
    sum_k1_k2 = 0
    m = len(S[0])
    for i in range(m):
        for j in range(m):
            sum_k1_k1 = K1[i][j] * K1[i][j]
            sum_k1_k2 = K1[i][j] * K2[i][j]
            sum_k2_k2 = K2[i][j] * K2[i][j]
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
        print("Calculating %d,%d" % (i,j))
        summa = 0
        for m in range(no_substrings):
            summa += s_doc_table[m,i] * s_doc_table[m,j]
        gram_matrix[i,j] = summa
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

    sub_Ds_table = np.zeros((len(S), len(arg[0])))
    for i, doc in enumerate(arg[0]):
        for j, K_ss in enumerate(arg[1]):
            time_start = time.time()

            K_Ds = ssk(doc[1], S[j], k, l)
            sub_Ds_table[j, i] = K_Ds / np.sqrt(DxD[doc[0]] * K_ss)

            time_end = time.time()
            average_total += time_end - time_start
            average_counter += 1.0

            if j % 500 == 0:
                print('\t\t' * thread_id + f"worker {thread_id} at: {j}")
        average_running = average_total / average_counter
        print(f"Worker {thread_id} finished document {doc[0]}.")
        print("Time: %d Average: %f s" % (time_end - time_start, average_running))

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

if __name__ == '__main__':
    entries = load_all_entries()[:10]
    all_bodies = [(x.id, x.clean_body) for x in entries]
    train_entries = [ (x.id, x.clean_body) for x in entries if x.lewis_split == 'TRAIN' ]
    substrings = create_all_substrings()


    #gram_matrix = approximate_matrix()
    #f = open('pickels/gram_matrix_all_features_100_first_documents_k_3_lambda_0_5.pkl', 'wb')
    #pickle.dump(gram_matrix,f)
    #f.close()

    SD_table = precompute_DxS_table(all_bodies, substrings, 3, 0.5, nworkers=3)
    f = open('pickels/s_doc_table_all_substrings_100_first_set_k_3_lambda_0_5.pkl', 'wb')
    pickle.dump(SD_table, f)
    f.close()
    # precalc_s_s(all_bodies, 3, 0.5)
