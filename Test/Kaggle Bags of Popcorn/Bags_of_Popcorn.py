# coding: utf-8

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import tree,ensemble
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")
from smart_sampling import smartSampling
from KaggleWord2VecUtility import KaggleWord2VecUtility  ## don't remove stopwords

data = pd.read_csv(os.path.join('labeledTrainData.tsv'), header=0,                 delimiter="\t", quoting=3)
print 'The first review is:'
print data["review"][0]

# Initialize an empty list to hold the clean reviews
clean_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
print "Cleaning and parsing the training set movie reviews...\n"
for i in xrange( 0, len(data["review"])):
    clean_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(data["review"][i], True)))
print clean_reviews[0]

nb_reviews = len(clean_reviews)
print nb_reviews
pop_size = 500
print 'keep only',pop_size

training_reviews = clean_reviews[:pop_size]
test_reviews = clean_reviews[pop_size:(2*pop_size)]

training_labels = np.asarray(data["sentiment"][:pop_size])
test_labels =  np.asarray(data["sentiment"][pop_size:(2*pop_size)])
print training_labels[:5]


def scoring_function(parameters):
	parameters = np.asarray( parameters , dtype=np.int32)
	nb_features,n_estimators = parameters
	nb_features = np.int(nb_features)
	n_estimators = np.int(n_estimators)
	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.
	vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = nb_features)

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of
	# strings.
	train_data_features = vectorizer.fit_transform(training_reviews)
	ch2 = SelectKBest(chi2, k=final_nb)
	ch2.fit(train_data_features,training_labels)
	chiSQ_val = ch2.scores_
	#print chiSQ_val[:10]
	index = np.argsort(chiSQ_val)[::-1]
	idx = index[:final_nb]
	test_data_features = vectorizer.transform(test_reviews)

	# Numpy arrays are easy to work with, so convert the result to an
	# array
	train_data_features = train_data_features.toarray()
	test_data_features = test_data_features.toarray()

	mean_res = 0.
	nb_try = 6
	for n_test in range(nb_try): 
		# Initialize a Random Forest classifier with n_estimators trees
		forest = RandomForestClassifier(n_estimators = n_estimators)
		forest.fit( train_data_features, training_labels )
		predicted_labels = forest.predict(test_data_features)

		# Compute accuracy
		res = 0
		for i in range(len(predicted_labels)):
			if (predicted_labels[i] == test_labels[i]):
				res += 1
		mean_res += (100. * np.float(res))/len(predicted_labels)

	return (mean_res/np.float(nb_try))


# In[7]:

### Fix parameters of the problem : ####

final_nb = 1000 ### the final number of bags kept
parameter_bounds = np.asarray( [[3000,15000],[50,1000]] )


#### EXP0 
nb_GCP_steps = 3
n_exp = 0

all_parameters,all_outputs = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
                                                  model = 'all',
                                                  nb_random_steps=3, n_clusters=1,verbose=True)

print all_outputs.shape

for i in range(all_outputs.shape[0]):
        np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/output_exp" +str(n_exp)+"_"+str(i)+".csv"),all_outputs[i], delimiter=",")
        np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/param_exp" +str(n_exp)+"_"+str(i)+".csv"),all_parameters[i], delimiter=",")

### EXP 2
n_exp = 4
print 'Starting exp',n_exp 
nb_GCP_steps = 80

all_parameters,all_outputs = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
                                                  #corr_kernel= 'squared_exponential',
                                                  model = 'all',
                                                  nb_random_steps=20, n_clusters=1, verbose=True)

print all_outputs.shape

for i in range(all_outputs.shape[0]):
        np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/output_exp" +str(n_exp)+"_"+str(i)+".csv"),all_outputs[i], delimiter=",")
        np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/param_exp" +str(n_exp)+"_"+str(i)+".csv"),all_parameters[i], delimiter=",")




