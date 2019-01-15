'''
Author: Arsalan Syed
Date: 15th Jan 2019
'''

import itertools
import numpy as np
import random
import ssk
import data
from sklearn import svm

def get_labels(entries,target_category):
	labels = []

	for entry in entries:
		if target_category in entry.topics:
			labels.append(1.0)
		else:
			labels.append(0.0)
		
	return labels
	
def get_data():
	#Create train / test set
	all_entries = data.load_all_entries()[0:num_docs]
	train_entries = [x for x in all_entries if x.lewis_split == "TRAIN"] 
	test_entries = [x for x in all_entries if x.lewis_split == "TEST"] 


	#Split input / output
	x_train = [x.clean_body for x in train_entries]
	y_train = get_labels(train_entries,target_category)

	x_test = [x.clean_body for x in test_entries]
	y_test = get_labels(test_entries,target_category)
	
	return x_train,y_train,x_test,y_test
	
#Constants
k=2
lam=0.5
num_docs = 2
target_category = "earn"

#Data
x_train,y_train,x_test,y_test = get_data()

gram_matrix_train = ssk.create_gram_matrix(x_train,k,lam,isNormalized = False)
gram_matrix_test = ssk.create_gram_matrix(x_test,k,lam,isNormalized = False)

#Create model and predict test data
svm_model = svm.SVC(kernel='precomputed')
svm_model.fit(gram_matrix_train,y_train)
predictions = svm_model.predict(gram_matrix_test)

#Evaluate predictions

