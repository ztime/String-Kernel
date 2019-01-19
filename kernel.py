import numpy as np
from approximation import load_s_doc_table_top_3000
from data import load_map_doc_id_to_index, load_all_entries, ReutersEntry
import os
import time
import pickle

PICKLES_PATH = './pickels/'
APPROX_KERNEL_PATH = './pickels/approx_kernels/'


def __finalize(X):
    n = X.shape[1]
    G = np.zeros((n,n))
    for i in range(n):
        t0 = time.time()
        for j in range(i, n):
            K = X[:,i] @ X[:,j].T
            G[i,j] = K
            G[j,i] = K
        print(f'at {i}/{n}. time:{(time.time()-t0)}s')
    return G

def __construct_kernel(top, k, _type='TRAIN'):
    print(f'Constructing approximation kernel, k={k}, top={top}.')

    # Load doc id to index mapping
    mapping = load_map_doc_id_to_index()

    # load all documents and find all included id's
    entries = load_all_entries()
    indices = [ mapping[x.id] for x in entries if x.lewis_split == _type ]

    # Load precomputed top 3000 s x all docs
    sD_table = load_s_doc_table_top_3000(k)[:top, indices]
    return __finalize(sD_table)



def __load_kernels(top, k):
    if os.path.isfile(f'{APPROX_KERNEL_PATH}TRAIN_KERNEL_k_{k}_top_{top}.pkl'):
        print('Training kernel and labels loaded.')
        with open(f'{APPROX_KERNEL_PATH}TRAIN_KERNEL_k_{k}_top_{top}.pkl', 'rb') as f:
            train_kernel = pickle.load(f)
    else:
        train_kernel = __construct_kernel(top, k)
        f = open(f'{APPROX_KERNEL_PATH}TRAIN_KERNEL_k_{k}_top_{top}.pkl', 'wb')
        pickle.dump(train_kernel, f)
        f.close()

    if os.path.isfile(f'{APPROX_KERNEL_PATH}TEST_KERNEL_k_{k}_top_{top}.pkl'):
        print('Testing kernel loaded.')
        with open(f'{APPROX_KERNEL_PATH}TEST_KERNEL_k_{k}_top_{top}.pkl', 'rb') as f:
            test_kernel = pickle.load(f)
    else:
        test_kernel = __construct_kernel(top, k, _type='TEST')
        f = open(f'{APPROX_KERNEL_PATH}TEST_KERNEL_k_{k}_top_{top}.pkl', 'wb')
        pickle.dump(test_kernel, f)
        f.close()
    return train_kernel, test_kernel


def __load_labels(topic, _type='TRAIN'):
    entries = load_all_entries()
    labels = [ 1.0 if topic in x.topics else 0.0 for x in entries if x.lewis_split == _type ]
    return np.array(labels)

def get_kernels_and_labels_top(top=3000, k=3, topic='earn'):
    assert k >= 3 and k <= 5
    assert top >= 1 and top <= 3000

    train_kernel, test_kernel = __load_kernels(top, k)
    train_labels = __load_labels(topic)
    test_labels = __load_labels(topic, _type='TEST')
    return train_kernel, train_labels, test_kernel, test_labels


if __name__ == '__main__':
    get_kernels_and_labels_top()
