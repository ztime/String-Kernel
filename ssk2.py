import unittest
import numpy as np

def get_all_indices_contain(t,x):
    indices = []
    index = 1
    for char in t:
        if char == x:
            indices.append(index)
        index += 1
    return indices


def calculate_K(s, t, k, lam, K_prime):
    if min(len(s), len(t)) < k:
        return 0

    K = 0
    x = s[-1]

    indices = get_all_indices_contain(t,x)

    summa = 0
    for index in indices:
        summa += K_prime[k-1][len(s[:-1])][index] * lam**2

    return calculate_K(s[:-1], t, k, lam, K_prime) + summa

def precalculate_K_prime(s, t, k, lam, K_prime, K_bis):
    for n in range(1, k):
        for i in range(len(s)+1):
            for j in range(len(t)+1):
                if min(len(s[:i]), len(t[:j])) < n:
                    continue

                if s[i-1] == t[j-1]:
                    K_bis[n][i][j] = lam * (K_bis[n][i][j-1] + lam * K_prime[n-1][i-1][j-1])
                else:
                    K_bis[n][i][j] = lam * K_bis[n][i][j]

                K_prime[n][i][j] = lam * K_prime[n][i-1][j] + K_bis[n][i][j]

    return K_prime

def ssk(s,t,k,lam):
    ''' K'0(s,t) = 1, for all s, t '''
    K_prime = np.zeros((k, len(s)+1, len(t)+1))
    K_prime[0,:,:] = np.ones((1, len(s)+1, len(t)+1))
    K_bis = np.zeros((k, len(s)+1, len(t)+1))

    precalculate_K_prime(s,t, k, lam, K_prime, K_bis)
    return calculate_K(s, t, k, lam, K_prime)



if __name__ == '__main__':
    s = "science is organized knowledge"
    t = "wisdom is organized life"
    k = 3
    lam = 0.4
    print(ssk(s, t, k, lam))
