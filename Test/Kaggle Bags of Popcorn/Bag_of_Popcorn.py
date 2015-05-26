n_exp = 705

# coding: utf-8

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import tree,ensemble
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score

import sys
sys.path.append("../../")
from smart_sampling import smartSampling
from KaggleWord2VecUtility import KaggleWord2VecUtility  ## don't remove stopwords

data = pd.read_csv(os.path.join('labeledTrainData.tsv'), header=0,                 delimiter="\t", quoting=3)
print 'The first review is:'
print data["review"][0]

if not os.path.exists("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/exp" +str(n_exp)):
	os.mkdir("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/exp" +str(n_exp))
else:
	print('Be carefull, directory already exists')

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

def scoring_function(parameters):
	pop_size, nb_features,n_estimators = parameters
    #print pop_size
    #training_reviews = clean_reviews[:pop_size]
    #test_reviews = clean_reviews[pop_size:(2*pop_size)]
    #training_labels = np.asarray(data["sentiment"][:pop_size])
    #test_labels =  np.asarray(data["sentiment"][pop_size:(2*pop_size)])
    	raw_X = clean_reviews[:(2*pop_size)]
    	Y = np.asarray(data["sentiment"][:(2*pop_size)])
    
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    	vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = nb_features)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    	train_data_features = vectorizer.fit_transform(raw_X)
    	ch2 = SelectKBest(chi2, k=final_nb)
    	ch2.fit(train_data_features,Y)
   	chiSQ_val = ch2.scores_
    #print chiSQ_val[:10]
    	index = np.argsort(chiSQ_val)[::-1]
    	idx = index[:final_nb]
    #test_data_features = vectorizer.transform(test_reviews)
    	X = train_data_features.toarray()
    	X = X[:,idx]
	forest = RandomForestClassifier(n_estimators = n_estimators)
    	cv_score = cross_val_score(forest,X,Y,cv=5)

    	return np.mean(cv_score)

### Fix parameters of the problem : ####
final_nb = 2000 ### the final number of bags kept
parameter_bounds = np.asarray( [[3000,15000],[50,1000]] )
data_size_bounds = [100,1000]

print 'Starting exp',n_exp

nb_GCP_steps = 100

#all_parameters,all_outputs = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
#                                                  #corr_kernel= 'squared_exponential',
#                                                  model = 'all',cluster_evol='variable',
#                                                  nb_random_steps=20, n_clusters=3, verbose=True)

all_parameters,all_outputs = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
                                            data_size_bounds = data_size_bounds,
                                            model = 'all', nb_parameter_sampling=2000,
                                            nb_random_steps=25,n_clusters=1,verbose=True)

print all_outputs.shape
print 'Exp',n_exp,'has just finished'

for i in range(all_outputs.shape[0]):
        np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/exp" +str(n_exp)+"/output_"+str(i)+".csv"),all_outputs[i], delimiter=",")
        np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/exp" +str(n_exp)+"/param_"+str(i)+".csv"),all_parameters[i], delimiter=",")


