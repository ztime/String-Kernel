'''
Author: Arsalan Syed
Date: 10th Jan 2019
'''

import itertools
import numpy as np

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

s="car"
t="cat"
k=2
lam=0.5
#can see that it obtains lambda^4 as kernel value
print(naive(s,t,k,lam),lam**4)