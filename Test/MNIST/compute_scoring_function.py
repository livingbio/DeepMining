n_computations = 25
pop_size = 5000


### import ####
import os
import numpy as np
from sklearn.cross_validation import cross_val_score,Bootstrap,KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from numpy import genfromtxt, savetxt

import random
import sys
sys.path.append("../../")
from smart_sampling import smartSampling


n_run = sys.argv[1]
print 'Arguments:',sys.argv


### Fix parameters : ####
### Fix parameters of the problem : ####
# nb_features, feat_select,max_ngram_range,max_df, min_df, alpha_NB, mode_tfidf
# pca_dim/50,degree,log10(gamma*1000)
parameter_bounds = np.asarray( [
        [5,14],
        [1,5],
        [0,4]] )

nb_GCP_steps = 0
nb_final_steps = 0



### set directory
if not os.path.exists("scoring_function/pop" +str(pop_size)):
    os.mkdir("scoring_function/pop" +str(pop_size))
else:
    print('Be carefull, directory already exists')

### get data
dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
X_data = np.asarray(train)
Y_data = np.asarray(target)

print X_data.shape,Y_data.shape


def scoring_function(parameters):
    subsample_idx = range(35000)
    random.shuffle(subsample_idx)
    subsample_idx = subsample_idx[:(pop_size)]
    #print subsample_idx
    subsample_data = X_data[subsample_idx,:]
    sub_Y = Y_data[subsample_idx]

    return scoring_function_cv(subsample_data,sub_Y,parameters)

def scoring_function_cv(subsample_data,Y,parameters):
    pca_dim,d,g = parameters
    pca_dim = 50 *pca_dim
    gamma = (10. ** g )/ 1000.
    
    pca = PCA(n_components = pca_dim)
    X = pca.fit_transform(subsample_data)
    kf = KFold(subsample_data.shape[0],n_folds=5)
    cv_results = []
    for train_idx,test_idx in kf:
        X_cv,Y_cv = X[train_idx,:],Y[train_idx]

        clf = SVC(kernel='poly',gamma=gamma,degree=d) #RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_cv,Y_cv)
        X_test = X[test_idx,:] 
        Y_pred = clf.predict(X_test)
        res = np.sum( Y_pred == Y[test_idx]) / float(Y_pred.shape[0])
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

f =open(("scoring_function/pop" +str(pop_size)+"/output_"+str(n_run)+".csv"),'w')
for line in all_outputs[0]:
    #for item in line:
    print>>f,line
    #print>>f,'\n'

np.savetxt(("scoring_function/pop" +str(pop_size)+"/param_"+str(n_run)+".csv"),all_parameters[0], delimiter=",")
