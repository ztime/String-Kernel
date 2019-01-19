import imp
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
import numpy
from kernel import get_kernels_and_labels_top
from data import ReutersEntry

def print_usage():
	print('usage: python svm.py <k> <top> <topic>\n' +
          '       where k any in {3, 4, 5}\n' +
          '       and S is either 1000 or 3000')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print_usage()
        sys.exit()

    k = int(sys.argv[1])
    top = int(sys.argv[2])
    topic = sys.argv[3]

    if k not in [3,4,5] or top not in [1000, 3000]:
        print_usage()

    # load kernels from disc.
    # (Will be computed if not on disc)
    train_kernel, train_labels, test_kernel, test_labels = get_kernels_and_labels_top(top, k, topic)

    # setup SVM.
    clf = SVC(C=1.0, # Penalty parameter C of the error term. Default 1.0
              kernel='precomputed', # set SVM to precomputed kernel
              shrinking=True,  # Whether to use the shrinking heuristic. Default true
              tol=0.001, # Tolerance for stopping criterion. Default 0.001
              cache_size=200 # Specify the size of the kernel cache (in MB). Default 200
            )

    # train model
    clf.fit(train_kernel, train_labels)

    # test model
    prediction = clf.predict(test_kernel, test_labels)

    # evaluate results
    print(f'Results from approximation kernel (k={k}, top {top}, topic {topic})')
    print(f'--------------------------------------------------------' + '-' * len(topic))
    print(f'Precision\t {precision_score(test_labels, prediction)}')
    print(f'recall   \t {recall_score(test_labels, prediction)}')
    print(f'f1       \t {f1_score(test_labels, prediction)}')
