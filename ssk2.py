import unittest
import numpy as np
import itertools
import time
from multiprocessing.dummy import Pool as ThreadPool


def get_all_indices_contain(t,x):
    indices = []
    index = 1
    for char in t:
        if char == x:
            indices.append(index)
        index += 1
    return indices


def calculate_K(s, t, k, lam, K_prime):
    K = 0
    for i in range(k, len(s)+1):
        x = s[i-1]
        indices = get_all_indices_contain(t,x)
        for index in indices:
            K += K_prime[k-1,len(s[:i-1]),index-1] * lam * lam

    return K

def precalculate_K_prime(s_string, t_string, k_in, lamb, K_prime, K_bis):
    length_s = len(s_string) + 1
    length_t = len(t_string) + 1
    for k in range(1, k_in):
        # for the upper and left edge of the K_prime matrix
        # the value is already precalculated as it will always be 0
        # so we start at 1 (for K_bis this also holds true but value
        # is 0)
        for s in range(1, length_s):
            for t in range(1, length_t):
                #from definition of K_prime
                if min(s,t) < k:
                    K_prime[k,s,t] = 0.0
                    #done here!
                    continue
                #calc K_bis
                # -1 because we need to use the empty string aswell
                if s_string[s-1] == t_string[t-1]:
                # if s_string[s] == t_string[t]:
                    K_bis[k,s,t] = lamb * ( K_bis[k,s,t-1] + lamb * K_prime[k-1,s-1,t-1] )
                else:
                    K_bis[k,s,t] = lamb * K_bis[k,s,t-1]
                K_prime[k,s,t] = lamb * K_prime[k,s-1,t] + K_bis[k,s,t]
    return K_prime

def ssk(s,t,k,lam):
    #Skapa k_p och k_b
    K_prime = np.zeros((k, len(s)+1, len(t)+1))
    K_bis = np.zeros((k, len(s)+1, len(t)+1))
    # K'0(s,t) = 1, for all s, t
    K_prime[0,:,:] = np.ones((1, len(s)+1, len(t)+1))

    precalculate_K_prime(s,t, k, lam, K_prime, K_bis)
    return calculate_K(s, t, k, lam, K_prime)

def ssk_tuple_args(tuple_args):
    x,y,s,t,k,lam = tuple_args
    #Skapa k_p och k_b
    #K_prime = np.zeros((k, len(s)+1, len(t)+1))
    K_prime = np.ones((k, len(s)+1, len(t)+1))
    K_bis = np.zeros((k, len(s)+1, len(t)+1))
    # K'0(s,t) = 1, for all s, t
    #K_prime[0,:,:] = np.ones((1, len(s)+1, len(t)+1))

    precalculate_K_prime(s,t, k, lam, K_prime, K_bis)
    calc_k = calculate_K(s, t, k, lam, K_prime)
    print("Calculated gram[%d,%d]" % (x,y))
    return calc_k

def normalized_ssk(s, t, k, lam):
    kernel_ss = ssk(s,s,k,lam)
    kernel_tt = ssk(t,t,k,lam)
    kernel_st = ssk(s,t,k,lam)
    return kernel_st / (np.sqrt(kernel_ss * kernel_tt))

# def create_gram_matrix_approximation(S, documents, k, lam):


'''
Creates a matrix where
       <----- documents ---->
       ^
       |
       |
       S
       |
       |

essentially [x,a] represents K(x,a) where
x is a document and a is a 3gram
'''
# def create_documents_s_matrix(S,documents, k, lam):
    # np.zeros()


def create_gram_matrix_from_documents(documents, k, lam):
    num_docs = len(documents)
    print("Calculating %dx%d gram matrix" % (num_docs, num_docs))
    entries = num_docs * num_docs
    gram = np.zeros((num_docs, num_docs))
    average_counter = 0.0
    average_running = 0.0
    average_total = 0.0
    for x,y in itertools.product(range(num_docs),range(num_docs)):
        start_time = time.time()
        gram[x,y] = ssk(documents[x], documents[y], k, lam)
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        average_counter += 1.0
        average_total += elapsed_seconds
        average_running = average_total / average_counter
        estimate = (entries - average_counter) * average_running
        unit_time = "s"
        if estimate > 3600:
            estimate /= 3600.0
            unit_time = "h"
        elif estimate > 60:
            estimate /= 60
            unit_time = "m"
        print("Calculated [%d,%d] (Average: %d seconds, estimited time left:%d%s)" % (x,y,average_running, estimate, unit_time))
    return gram

if __name__ == '__main__':
    #Test
    s = "science is organized knowledge"
    t = "wisdom is organized life"
    for l in [ x / 10.0 for x in range(1,11,1)]:
        print("----------------------------")
        for k in range(1, 7):
            print("K_%d l = %f: %f" % (k, l, normalized_ssk(s, t, k, l)))
    print("-----------------------------------")
    s = "cat"
    t = "rca"
    print("K_%d l = %f: %f" % (2, 0.5, ssk(s, t, 2, 0.5)))
    asd
    #Testing 100x100
    from data import load_all_entries, ReutersEntry
    import sys
    sys.setrecursionlimit(4000)
    all_entries = load_all_entries()
    first_100 = [ x.clean_body for x in all_entries[:2] ]
    gram = create_gram_matrix_from_documents(first_100, 3, 0.5)
    # gram = create_gram_matrix_from_documents(['abc', first_100, 3, 0.5)
