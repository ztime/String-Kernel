import imp
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from data import ReutersEntry
import data
import numpy



def train(train_kernel, train_labels):
    clf = SVC(C=1.0, # Penalty parameter C of the error term. Default 1.0
                kernel='precomputed', # set SVM to precomputed kernel
                shrinking=True,  # Whether to use the shrinking heuristic. Default true
                tol=0.001, # Tolerance for stopping criterion. Default 0.001
                cache_size=200 # Specify the size of the kernel cache (in MB). Default 200
                )
    clf.fit(train_kernel, train_labels)
    return clf


def classify(kernel, labels):
    
    predictions = clf.predict(kernel)

    f1 = f1_score(labels, predictions, average=None)
    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)
    return f1,precision,recall
    

def print_usage():
	print('usage: python svm.py <kernel>')


def get_labels_for_category(reuter_entries, target_categories):
    labels = []

    for entry in reuter_entries:
        for category in target_categories:
            if category in entry.topics:
                labels.append(float(target_categories.index(category)))
                break
        labels.append(len(target_categories)) #need label if none of the above
            
    return labels


def get_labels(target_categories,num_docs):
    
    #Create train / test set
	all_entries = data.load_all_entries()
	train_entries = [x for x in all_entries if x.lewis_split == "TRAIN"] 
	test_entries = [x for x in all_entries if x.lewis_split == "TEST"] 

	train_labels = get_labels_for_category(train_entries,target_categories)[0:num_docs]
	test_labels = get_labels_for_category(test_entries,target_categories)[0:num_docs]
	
	return train_labels, test_labels 

    
   
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()

    kernel_type = sys.argv[1]
    num_docs = 100
    
    if kernel_type == "ssk":
        train_kernel = numpy.load("pickels/100_doc_gram_matrix")
        test_kernel = numpy.load("pickels/<TEST_KERNEL_FILENAME>")
    elif kernel_type == "approx":
        train_kernel = numpy.load("pickels/<TRAIN_APPROX>")
        test_kernel = numpy.load("pickels/<TEST_APPROX>")
        
    target_categories = ["earn","acquisition","crude","corn"]
    train_labels, test_labels = get_labels(target_categories, num_docs)

	# Obtain SVM model
    clf = train(train_kernel, train_labels)		
         
	# Do prediction on test data
    f1,precision,recall = classify(test_kernel, test_labels)
    
    print("f1",f1)
    print("Precision",precision)
    print("Recall",recall)
	