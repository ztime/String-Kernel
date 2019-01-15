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
    summa = 0.0
    for index in indices:
        summa += K_prime[k-1][len(s[:-1])][index] * lam**2

    return calculate_K(s[:-1], t, k, lam, K_prime) + summa

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
                    K_prime[k,s,t] = 0
                    #done here!
                    continue
                #calc K_bis
                # -1 because we need to use the empty string aswell
                if s_string[s-1] == t_string[t-1]:
                    K_bis[k,s,t] = lamb * ( K_bis[k,s-1,t-1] + (lamb * K_prime[k-1,s,t]) ) 
                else:
                    K_bis[k,s,t] = lamb * K_prime[k,s-1,t-1] + K_bis[k,s,t-1]
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

def normalized_ssk(s, t, k, lam):
    kernel_ss = ssk(s,s,k,lam)
    kernel_tt = ssk(t,t,k,lam)
    kernel_st = ssk(s,t,k,lam)
    return kernel_st / (np.sqrt(kernel_ss * kernel_tt))


if __name__ == '__main__':
    s = "science is organized knowledge"
    t = "wisdom is organized life"
    lam = 0.5
    for k in range(1, 7): 
        print("K_%d: %f" % (k, normalized_ssk(s, t, k, lam)))
