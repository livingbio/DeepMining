n_computations = 30
pop_size = 5000
dir_name = "v4_pop" + str(pop_size)

### import ####
import os
import numpy as np
from sklearn.cross_validation import cross_val_score,KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import genfromtxt, savetxt

import random
import sys
sys.path.append("../../")
from smart_sampling import smartSampling


n_run = sys.argv[1]
print 'Arguments:',sys.argv


### Fix parameters : ####
# blur_ksize,blur_sigma,pca_dim/10,degree,log10(gamma*1000)
parameter_bounds = np.asarray( [
        [0,2],
        [0,4],
        [1,37],
        [1,5],
        [0,4]] )

nb_GCP_steps = 0
nb_final_steps = 0



### set directory
if not os.path.exists("scoring_function/"+dir_name):
    os.mkdir("scoring_function/"+dir_name)
else:
    print('Be carefull, directory already exists')

### data idx :
# 0 : without denoising / blurring
# 1 : GB_1_1_1
# 2 : GB_1_1_2
# 3 : GB_1_1_3
# 4 : GB_3_3_1
# 5 : GB_3_3_2
# 6 : GB_3_3_3
data_filenames = ["train","train_clean_GB_1_1_1","train_clean_GB_1_1_2","train_clean_GB_1_1_3", \
                  "train_clean_GB_3_3_1","train_clean_GB_3_3_2","train_clean_GB_3_3_3"]

datasets = []
d0 = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
Y_data = np.asarray([x[0] for x in d0])[:20000]

for fn in data_filenames:
    all_pca_files = []
    for pca_d in range(parameter_bounds[2][0],parameter_bounds[2][1]):
        X_fn = genfromtxt("data/"+fn + "_pca_"+str(pca_d)+".csv",delimiter=',')
        all_pca_files.append(X_fn)
    print 'Loaded',fn
    datasets.append( all_pca_files )


def scoring_function(parameters):
    blur_ksize,blur_sigma,pca_dim,_,_ = parameters

    if(blur_sigma == 0):
        dataset_idx = 0
    else:
        dataset_idx = 3*blur_ksize + blur_sigma

    X_data = datasets[dataset_idx][pca_dim-1]

    subsample_idx = range(20000)
    random.shuffle(subsample_idx)
    subsample_idx = subsample_idx[:(pop_size)]
    
    subsample_data = X_data[subsample_idx,:]
    sub_Y = Y_data[subsample_idx]

    return scoring_function_cv(subsample_data,sub_Y,parameters)

def scoring_function_cv(X,Y,parameters):
    _,_,_,d,g = parameters
    gamma = (10. ** g )/ 1000.

    kf = KFold(pop_size,n_folds=5)
    cv_results = []
    for train_idx,test_idx in kf:
        X_cv,Y_cv = X[train_idx,:],Y[train_idx]

        clf = SVC(kernel='poly',gamma=gamma,degree=d) #,coef0=coef0) 
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
                                            nb_iter_final = 0, returnAllParameters=False,
                                            nb_random_steps=n_computations,
                                            n_clusters=1,verbose=True)

#print all_outputs

f =open(("scoring_function/"+dir_name+"/output_"+str(n_run)+".csv"),'w')
for line in all_outputs[0]:
    print>>f,line

np.savetxt(("scoring_function/"+dir_name+"/param_"+str(n_run)+".csv"),all_parameters[0], delimiter=",")
