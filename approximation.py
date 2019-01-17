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
from multiprocessing.dummy import Pool as ThreadPool

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

def compute_s_doc_table(S,documents,k,l):
    # contains K(x,s) for all documents and substrings
    average_counter = 0.0
    average_total = 0.0
    average_running = 0.0
    s_doc_table = np.zeros((len(S), len(documents)))
    sub_precomputed = dict()
    s_s = load_precalc_s_s()
    for doc in range(len(documents)):
        doc_id = documents[doc][0]
        doc_content = documents[doc][1]
        print("Calculating for document %d" % doc)
        time_start = time.time()
        # ss_ssk = ssk(documents[doc],documents[doc],k,l)
        ss_ssk = s_s[doc_id]
        time_end = time.time()
        average_total += time_end - time_start
        for s in range(len(S)):
            time_start = time.time()
            if S[s] not in sub_precomputed:
                sub_precomputed[S[s]] = ssk(S[s],S[s],k,l)
            # s_doc_table[s,doc] = ssk(doc_content,S[s],k,l)
            st = ssk(doc_content,S[s],k,l)
            s_doc_table[s,doc] = st / np.sqrt(ss_ssk * sub_precomputed[S[s]])
            time_end = time.time()
            average_counter += 1.0
            average_total += time_end - time_start
            if s % 500 == 0:
                print("Currently: %d" % s)
        average_running = average_total / average_counter
        print("Time: %d Average: %f s" % (time_end - time_start, average_running))
    return s_doc_table

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

def load_precalc_s_s():
    with open('pickels/s_s_k_3_l_0_5_all_documents.pkl', 'rb') as f:
        s_s = pickle.load(f)
    return s_s

def load_s_doc_table():
    f = open('pickels/s_doc_table_all_substrings_100_first_set_k_3_lambda_0_5.pkl', 'rb')
    s_doc = pickle.load(f)
    return s_doc

if __name__ == '__main__':
    entries = load_all_entries()
    train_entries = [ (x.id, x.clean_body) for x in entries if x.lewis_split == 'TRAIN' ]
    substrings = create_all_substrings()

    gram_matrix = approximate_matrix()
    f = open('pickels/gram_matrix_all_features_100_first_documents_k_3_lambda_0_5_NORMALIZED.pkl', 'wb')
    pickle.dump(gram_matrix,f)
    f.close()

    # s_doc_table = compute_s_doc_table(substrings, all_bodies[:100], 3, 0.5)
    # f = open('pickels/s_doc_table_all_substrings_100_first_set_k_3_lambda_0_5.pkl', 'wb')
    # pickle.dump(s_doc_table, f)
    # f.close()
    # precalc_s_s(all_bodies, 3, 0.5)

