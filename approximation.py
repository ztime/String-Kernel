import sys
import numpy as np
import math
from operator import itemgetter
from data import load_all_entries, ReutersEntry, ALLOWED_CHARS, SAVE_FOLDER
from ssk import normalized_ssk, ssk
from enum import Enum
import time
import pickle

class FileEnum(Enum):
    AFFIX = ".s_doc_table.pkl"
    K_3_100_FIRST_ALL_S = "k_3_100_first_docs_all_substrings"
    DOC_DOC_SSK_K_3 = "doc_doc_ssk_k_3_l_0_5"


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
def approximate_matrix(S, documents, k, l):
    s_doc_table = compute_s_doc_table(S,documents,k,l)
    path = "%s/%s%s" % (SAVE_FOLDER, FileEnum.K_3_100_FIRST_ALL_S,FileEnum.AFFIX)
    f = open(path, 'wb')
    pickle.dump(s_doc_table, f)
    f.close()

def compute_s_doc_table(S,documents,k,l):
    # contains K(x,s) for all documents and substrings
    average_counter = 0.0
    average_total = 0.0
    average_running = 0.0
    s_doc_table = np.zeros((len(S), len(documents)))
    sub_precomputed = dict()
    for doc in range(len(documents)):
        print("Calculating for document %d" % doc)
        time_start = time.time()
        ss_ssk = ssk(documents[doc],documents[doc],k,l)
        time_end = time.time()
        average_total += time_end - time_start
        for s in range(len(S)):
            time_start = time.time()
            if S[s] not in sub_precomputed:
                sub_precomputed[S[s]] = ssk(S[s],S[s],k,l)
            s_doc_table[s,doc] = ssk(documents[doc],S[s],k,l)
            st = ssk(documents[doc],S[s],k,l)
            s_doc_table[s,doc] = st / np.sqrt(ss_ssk * sub_precomputed[S[s]])
            time_end = time.time()
            average_counter += 1.0
            average_total += time_end - time_start
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

if __name__ == '__main__':
    entries = load_all_entries()
    first_100 = [ x.clean_body for x in entries ]
    substrings = create_all_substrings()
    approximate_matrix(substrings[:50], first_100[1:5], 3, 0.5)

