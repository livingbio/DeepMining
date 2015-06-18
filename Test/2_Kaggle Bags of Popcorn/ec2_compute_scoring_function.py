n_run = 0
n_computations = 1000
pop_size = 5000


### import ####
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score,Bootstrap,KFold
from matplotlib import cm
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2

import random
import sys
sys.path.append("../../")
from smart_sampling import smartSampling
from KaggleWord2VecUtility import KaggleWord2VecUtility  ## don't remove stopwords


### Fix parameters : ####
### Fix parameters of the problem : ####
# nb_features, feat_select,max_ngram_range,max_df, min_df, alpha_NB, mode_tfidf
parameter_bounds = np.asarray( [
        [13,33],
        [5,10],
        [1,5],
        [4,11],
        [0,2],
        [1,11],
        [1,4]] )

nb_GCP_steps = 0
nb_final_steps = 0



### set directory
if not os.path.exists("scoring_function/ft-select_pop" +str(pop_size)):
    os.mkdir("scoring_function/ft-select_pop" +str(pop_size))
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
    subsample_idx = range(20000)
    random.shuffle(subsample_idx)
    subsample_idx = subsample_idx[:(pop_size)]
    #print subsample_idx
    subsample_clean_reviews = [clean_reviews[i] for i in subsample_idx]
    sub_Y = np.asarray([data["sentiment"][i] for i in subsample_idx] )

    return scoring_function_cv(subsample_clean_reviews,sub_Y,parameters)


def scoring_function_cv(subsample_clean_reviews,Y,parameters):
    nb_features, feat_select,max_ngram_range,max_df, min_df, alpha_NB,mode_tfidf = parameters

    nb_features = 1000 * nb_features
    min_ngram_range = 1
    max_df = max_df / 10.
    min_df = min_df / 10.
    alpha_NB = alpha_NB / 10.
    
    use_idf = True
    tfidf_norm = 'l2'
    sub_tf = False
    if(mode_tfidf == 1):
        tfidf_norm = 'l1'
    if(mode_tfidf == 3):
        sub_tf = True

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = nb_features,
                             ngram_range= (min_ngram_range,max_ngram_range),
                             max_df = max_df ,
                             min_df = min_df )

    hist = vectorizer.fit_transform(subsample_clean_reviews)
    BOF_features = hist.toarray()
    
    tfidf = TfidfTransformer(norm=tfidf_norm, use_idf=use_idf, smooth_idf=True, sublinear_tf=sub_tf)
    X = tfidf.fit_transform(BOF_features)
    final_nb_features = feat_select * X.shape[1] / 10
    print final_nb_features

    n_reviews = len(subsample_clean_reviews)
    ### CV ###
    kf = KFold(n_reviews,n_folds=5)
    cv_results = []
    for train_idx,test_idx in kf:
        X_cv,Y_cv = X[train_idx,:],Y[train_idx]
        
        ch2 = SelectKBest(chi2, k=final_nb_features)
        ch2.fit(X_cv,Y_cv)
        chiSQ_val = ch2.scores_
        index = np.argsort(chiSQ_val)[::-1]
        best_feat = index[:final_nb_features]
        X_cv = X_cv[:,best_feat]

        clf = MultinomialNB(alpha = alpha_NB)
        clf.fit(X_cv,Y_cv)
        X_test = X[test_idx,:]
        Y_pred = clf.predict_proba(X_test[:,best_feat])[:,1]
        res = roc_auc_score(Y[test_idx],Y_pred)
        cv_results.append(res)
    print 'CV res :',np.mean(cv_results)
    cv_std = np.std(cv_results)
    print 'STD:',cv_std

    return cv_results


print 'Start for run',n_run
print 'Go for',n_computations,'computations \n'

all_parameters,all_outputs = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,isInt=True,
                                            model = 'random', nb_parameter_sampling=500,
                                            nb_iter_final = 0, returnAllParameters=False, # just when computing the scoring function values
                                            nb_random_steps=n_computations,
                                            n_clusters=1,verbose=True)

#print all_outputs

f =open(("scoring_function/ft-select_pop" +str(pop_size)+"/output_"+str(n_run)+".csv"),'w')
for line in all_outputs[0]:
    #for item in line:
    print>>f,line
    #print>>f,'\n'

np.savetxt(("scoring_function/ft-select_pop" +str(pop_size)+"/param_"+str(n_run)+".csv"),all_parameters[0], delimiter=",")
