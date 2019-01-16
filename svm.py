import imp
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
import sys



def classify(kernel, labels):
    clf = SVC(C=1.0, # Penalty parameter C of the error term. Default 1.0
              kernel='precomputed', # set SVM to precomputed kernel
              shrinking=True, # Whether to use the shrinking heuristic. Default true
              tol=0.001, # Tolerance for stopping criterion. Default 0.001
              cache_size=200, # Specify the size of the kernel cache (in MB). Default 200
             )

    clf.fit(kernel, labels)
    predictions = clf.predict(kernel)

    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python svm.py <kernel>')

    # TODO: load pre-computed kernel
    classify(kernel, labels)
