n_exp = 2001

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import tree,ensemble
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from sklearn.cross_validation import Bootstrap,KFold
import random

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


### Fix parameters of the problem : ####
parameter_bounds = np.asarray( [[1000,15000],[50,1000]] )
nb_GCP_steps = 70
pop_size = 5000   
data_size_bounds = [pop_size,pop_size] ## constant here !


def scoring_function(parameters):
    subsample_idx = range(25000)
    random.shuffle(subsample_idx)
    subsample_idx = subsample_idx[:(2*pop_size)]
    #print subsample_idx
    subsample_clean_reviews = [clean_reviews[i] for i in subsample_idx]
    sub_Y = np.asarray([data["sentiment"][i] for i in subsample_idx] )

    return scoring_function_cv(subsample_clean_reviews,sub_Y,parameters)


def scoring_function_cv(subsample_clean_reviews,Y,parameters):
    nb_features,n_estimators = parameters

    vectorizer = CountVectorizer(analyzer = "word",
                                tokenizer = None,          
                                preprocessor = None,
                                stop_words = 'english', 
                                max_features = nb_features)

    hist = vectorizer.fit_transform(subsample_clean_reviews)
    train_data_features = hist.toarray()

    ### NO feature selection here ###
    #ch2 = SelectKBest(chi2, k=final_nb)
    #ch2.fit(train_data_features,Y)
    #chiSQ_val = ch2.scores_
    #print chiSQ_val[:10]
    #index = np.argsort(chiSQ_val)[::-1]
    #idx = index[:final_nb]
    
    X = train_data_features #[:,idx]
    n_reviews = len(subsample_clean_reviews)
    
    ### CV ###
    kf = KFold(n_reviews,n_folds=5)
    cv_results = []
    for train_idx,test_idx in kf:
        X_cv,Y_cv = X[train_idx,:],Y[train_idx]
        forest = RandomForestClassifier(n_estimators = n_estimators)
        forest.fit(X_cv,Y_cv)
        Y_pred = forest.predict(X[test_idx,:])
        res = np.sum(Y_pred == Y[test_idx])/(n_reviews/5.)
        cv_results.append(res)
    print 'CV res :',cv_results
    cv_std = np.std(cv_results)
    print 'STD:',cv_std

    return np.mean(cv_results)-0.5*cv_std

print 'Start exp',n_exp

all_parameters,all_raw_outputs,all_mean_outputs, all_std_outputs = \
    smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
                                            #data_size_bounds = data_size_bounds,
                                            model = 'GCP', nb_parameter_sampling=2000,
                                            nb_random_steps=30, n_clusters=1,verbose=True,
                                            acquisition_function = 'EI')


print 'Exp',n_exp,'has just finished'


for i in range(all_outputs.shape[0]):
    f =open(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/exp" +str(n_exp)+"/output_"+str(i)+".csv"),'w')
    for line in all_raw_outputs[i]:
        print>>f,line
    f.close()
    np.savetxt(("/afs/csail.mit.edu/u/s/sdubois/DeepMining/Test/Kaggle Bags of Popcorn/exp_results/exp" +str(n_exp)+"/param_"+str(i)+".csv"),all_parameters[i], delimiter=",")
