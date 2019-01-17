import unittest
import numpy as np
import itertools
import time
from data import load_all_entries, ReutersEntry
import pickle
import os
import sys

FULL_STRING_PATH = './pickels'

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

def ssk(s,t,k,lam, K_prime=None):
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

def normalized_ssk(s, t, k, lam, K_primes=[None] * 3):
    kernel_ss = ssk(s,s,k,lam, K_primes[0])
    kernel_tt = ssk(t,t,k,lam, K_primes[1])
    kernel_st = ssk(s,t,k,lam, K_primes[2])
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
def calculate_Gram_matrix_from_docs(documents, k, lam=0.5, normalized=True):
	num_docs = len(documents)
	print(f'Constructing Gram matrix of {num_docs}x{num_docs} documents (k={k})')
	entries = num_docs * num_docs
	average_counter = 0.0
	average_running = 0.0
	average_total = 0.0

	GRAM = np.zeros((num_docs, num_docs))
	SSK = {}

	for x,y in itertools.product(range(num_docs),range(num_docs)):
		start_time = time.time()

		s, t = documents[x].clean_body, documents[y].clean_body
		if normalized:
			K_st = ssk(s, t, k, lam)

			ss_cache_name = f'({documents[x].id}:{documents[x].id})'
			tt_cache_name = f'({documents[y].id}:{documents[y].id})'

			if ss_cache_name not in SSK:
				SSK[ss_cache_name] = ssk(s, s, k, lam)
			K_ss = SSK[ss_cache_name]

			if tt_cache_name not in SSK:
				SSK[tt_cache_name] = ssk(t, t, k, lam)
			K_tt = SSK[tt_cache_name]

			GRAM[x, y] = K_st / (np.sqrt(K_ss * K_tt))
		else:
			GRAM[x, y] = K_st = ssk(s, t, k, lam)

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
	#store in pkl file
	save_kernel(f'100_doc_gram_matrix', GRAM)

def save_kernel(filename, data):
	file = open(f'{FULL_STRING_PATH}/{filename}', 'wb')
	pickle.dump(data, file)
	file.close()

def load_kernel(filename):
	if os.path.isfile(f'{FULL_STRING_PATH}/{filename}'):
		with open(f'{FULL_STRING_PATH}/{filename}', 'rb') as f:
			K_prime = pickle.load(f)
	if not isinstance(K_prime, np.ndarray):
		print(f'PKL not found: {filename}')
		print('Force quit..')
		sys.exit()
	return K_prime

if __name__ == '__main__':

	# load first 100 documents
	docs = load_all_entries()[:1]
	lamb = 0.5
	k = 3
	calculate_Gram_matrix_from_docs(docs, k, lamb)
