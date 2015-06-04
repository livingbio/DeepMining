n_words = 500
n_run = 0
n_computations = 1000
pop_size = 500   


### import ####
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


### Fix parameters : ####
parameter_bounds = np.asarray( [[n_words,n_words+99],[10,100]] )
nb_GCP_steps = 0
nb_final_steps = 0



### set directory
if not os.path.exists("scoring_function/words" +str(n_words)):
    os.mkdir("scoring_function/words" +str(n_words))
else:
    print('Be carefull, directory already exists')

### get data
data = pd.read_csv(os.path.join('labeledTrainData.tsv'), header=0,
                  delimiter="\t", quoting=3)
print 'The first review is:'
print data["review"][0]

clean_reviews = []
print "Cleaning and parsing the training set movie reviews...\n"
for i in xrange( 0, len(data["review"])):
    clean_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(data["review"][i], True)))
print clean_reviews[0]

nb_reviews = len(clean_reviews)
print nb_reviews


def scoring_function(parameters):
    subsample_idx = range(25000)
    random.shuffle(subsample_idx)
    subsample_idx = subsample_idx[:(2*pop_size)]
    subsample_clean_reviews = [clean_reviews[i] for i in subsample_idx]
    sub_Y = np.asarray([data["sentiment"][i] for i in subsample_idx] )

    return scoring_function_cv(subsample_clean_reviews,sub_Y,parameters)


def scoring_function_cv(subsample_clean_reviews,Y,parameters):
    nb_features,n_estimators = parameters
    nb_features = 10*nb_features
    n_estimators = 10*n_estimators
    
    vectorizer = CountVectorizer(analyzer = "word",
                                tokenizer = None,          
                                preprocessor = None,
                                stop_words = 'english', 
                                max_features = nb_features)

    hist = vectorizer.fit_transform(subsample_clean_reviews)
    train_data_features = hist.toarray()

    X = train_data_features
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
    print 'CV res :',np.mean(cv_results)

    return cv_results


print 'Start for ',n_words,'words'
print 'Go for',n_computations,'computations \n'

all_parameters,all_outputs = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
                                            model = 'random', nb_parameter_sampling=2000,
                                            nb_iter_final = 0, returnAllParameters=False, # just when computing the scoring function values
                                            nb_random_steps=n_computations,
                                            n_clusters=1,verbose=True)

#print all_outputs

f =open(("scoring_function/words" +str(n_words)+"/output_"+str(n_run)+".csv"),'w')
for line in all_outputs[0]:
    #for item in line:
    print>>f,line
    #print>>f,'\n'

np.savetxt(("scoring_function/words" +str(n_words)+"/param_"+str(n_run)+".csv"),all_parameters[0], delimiter=",")
