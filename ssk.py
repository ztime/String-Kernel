'''
Author: Arsalan Syed
Date: 10th Jan 2019
'''

import itertools
import numpy as np
import unittest

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

'''
Given string t and char x, finds all indices
where x appears in t. Note that indexing is 
counted starting at 1
'''
def get_all_indices_contain(t,x):
	indices = []
	index = 1
	for char in t:
		if char == x:
			indices.append(index)
		index += 1
	return indices

def calc_k_prime(s,t,i,lam,kprime_table):
	if kprime_table[i][len(s)][len(t)] > 0:
		return kprime_table[i][len(s)][len(t)]

	if i==0:
		kprime_table[i][len(s)][len(t)] = 1
		return kprime_table[i][len(s)][len(t)]
	elif min(len(s),len(t)) < i:
		kprime_table[i][len(s)][len(t)] = 0
		return kprime_table[i][len(s)][len(t)]
	
	summation = 0
	x=s[-1]
	indices = get_all_indices_contain(t,x)
	for j in indices:
		summation += calc_k_prime(s[:-1],t[0:j-1],i-1,lam,kprime_table)*np.power(lam,len(t)-j+2)
	
	kprime_table[i][len(s)][len(t)] = lam*calc_k_prime(s[:-1],t,i,lam,kprime_table)+summation
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
	
	

class Test(unittest.TestCase):

	def test_1(self):
		s="car"
		t="cat"
		k=2
		lam=0.5
		self.assertEqual(ssk(s,t,k,lam),lam**4)


if __name__ == '__main__':
	unittest.main()
