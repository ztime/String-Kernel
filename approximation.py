import sys
import numpy as np
import math
from operator import itemgetter
from data import load_all_entries, ReutersEntry


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

if __name__ == '__main__':
    entries = load_all_entries()
    print(contiguos_from_first_100_entries(entries))

