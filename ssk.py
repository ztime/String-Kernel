'''
Author: Arsalan Syed
Date: 10th Jan 2019
'''

import itertools
import numpy as np
import unittest
from data import load_all_entries, ReutersEntry
import sys

sys.setrecursionlimit(4000)

def find_indices_with_substring(s,k,substring):
	start = 0
	end = start+k
	valid_indices = []
	for i in range(len(s)-k+1):
		if s[start:end] == substring:
			valid_indices.append((start,end))
		start += 1
		end += 1
	return valid_indices

def index_length(indices):
	return indices[1]-indices[0]

def get_alphabet():
	alphabet = []
	for c in range(97,123):
		alphabet.append(chr(c))
	alphabet.append(' ')
	return alphabet

alphabet = get_alphabet()

'''
s,t are strings
k is int
lam: lambda (decay factor)

Computes the kernel (SSK) using the naive algorithm
'''
def naive(s,t,k,lam):
	kernel_sum = 0
	finite_strings = [''.join(x) for x in itertools.permutations(alphabet,k)]

	for substring in finite_strings:
		s_indices = find_indices_with_substring(s,k,substring)
		t_indices = find_indices_with_substring(t,k,substring)

		for s_index in s_indices:
			for t_index in t_indices:

				kernel_sum += np.power(lam,index_length(s_index)+index_length(t_index))

	return kernel_sum


storage = {}

'''
Given string t and char x, finds all indices
where x appears in t. Note that indexing is
counted starting at 1
'''
def get_all_indices_contain(t,x):
	key = t + ' ' + x

	if key not in storage:
		indices = []
		index = 1
		for char in t:
			if char == x:
				indices.append(index)
			index += 1
		storage[key] = indices
		return indices
	else:
		return storage[key]


def calc_k_bis(s,t,i,lam, kprime_table):
	summation = 0
	x=s[-1]
	indices = get_all_indices_contain(t,x)
	for j in indices:
		summation += calc_k_prime(s[:-1],t[0:j-1],i-1,lam,kprime_table)*np.power(lam,len(t)-j+2)
	return summation

def calc_k_prime(s,t,i,lam,kprime_table):
	if kprime_table[i][len(s)][len(t)] > 0:
		return kprime_table[i][len(s)][len(t)]

	if i==0:
		kprime_table[i][len(s)][len(t)] = 1
		return kprime_table[i][len(s)][len(t)]
	elif min(len(s),len(t)) < i:
		kprime_table[i][len(s)][len(t)] = 0
		return kprime_table[i][len(s)][len(t)]


	#print(t, x, indices)

	kprime_table[i][len(s)][len(t)] = lam*calc_k_prime(s[:-1],t,i,lam,kprime_table)+calc_k_bis(s,t,i,lam,kprime_table)
	return kprime_table[i][len(s)][len(t)]

def calc_k(s,t,i,lam,k_table,kprime_table):
	if k_table[i][len(s)][len(t)] > 0:
		return k_table[i][len(s)][len(t)]

	if min(len(s),len(t)) < i:
		k_table[i][len(s)][len(t)] = 0
		return k_table[i][len(s)][len(t)]

	summation = 0
	x=s[-1]
	indices = get_all_indices_contain(t,x)
	for j in indices:
		summation += calc_k_prime(s[:-1],t[0:j-1],i-1,lam,kprime_table)*np.power(lam,2)

	k_table[i][len(s)][len(t)] = calc_k(s[:-1],t,i,lam,k_table,kprime_table)+summation
	return k_table[i][len(s)][len(t)]

def ssk(s,t,k,lam):
	k_table = np.zeros((k+1,len(s)+1,len(t)+1))
	kprime_table = np.zeros((k+1,len(s)+1,len(t)+1))
	return calc_k(s,t,k,lam,k_table,kprime_table)

def ssk_normalized(s,t,k,lam):
	kernel_st = ssk(s,t,k,lam)
	kernel_ss = ssk(s,s,k,lam)
	kernel_tt = ssk(t,t,k,lam)
	return kernel_st/(np.sqrt(kernel_ss*kernel_tt))

def create_gram_matrix(documents,k,lam,isNormalized = True):
	n = len(documents)
	G = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if i <= j: #K(s,t) = K(t,s), only want triangular matrix
				if isNormalized:
					G[i][j] = ssk_normalized(documents[i],documents[j],k,lam)
				else:
					G[i][j] = ssk(documents[i],documents[j],k,lam)
	return G

class Test(unittest.TestCase):

	def test_1(self):
		s="science is organized knowlage"
		t="wisdom is organized life"
		k=2
		lam=0.5
		print(ssk_normalized(s,t,k,lam))
		#	self.assertEqual(ssk(s,t,k,lam),lam**4)


if __name__ == '__main__':
	s="science is organized knowledge"
	t="wisdom is organized life"
	k=2
	lam=0.5

	print(ssk_normalized(s,t,k,lam))
	#unittest.main()
