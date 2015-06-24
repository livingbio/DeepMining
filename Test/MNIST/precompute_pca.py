data_size = 20000

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

filename = 'train_clean_GB_1_1_2'

pca_dim_boundsL,pca_dim_boundsU =  1,36 #int(sys.argv[1]),int(sys.argv[2])
print 'Arguments:',sys.argv
pca_range = range(pca_dim_boundsL,pca_dim_boundsU+1)
print 'PCA range:',pca_range

### get data
#dataset = genfromtxt(open(filename+'.csv','r'), delimiter=',', dtype='f8')[1:]    
#target = [x[0] for x in dataset]
#train = [x[1:] for x in dataset]
X_data = genfromtxt(filename+'.csv',delimiter=',')
#Y_data = np.asarray(target)[:data_size]

print X_data.shape

for d in pca_range:
    pca_dim = 10*d
    pca = PCA(n_components = pca_dim)
    X = pca.fit_transform(X_data)
    print X.shape
    np.savetxt("data/"+filename + "_pca_" +str(d),X, delimiter=",")
