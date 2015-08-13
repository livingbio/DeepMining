import numpy as np
import sys
sys.path.append("../../")
from run_experiment import runExperiment

n_exp = 1
first_exp = 5001 # int(sys.argv[1])

print 'Arguments:',sys.argv


###  Parameters  ###
GCP_mapWithNoise= False
GCP_useAllNoisyY= False
model_noise = 'EGN'
model = 'GCP'
n_candidates= 100
n_random_init= 10
n_iter = 30
n_clusters = 1
cluster_evol = 'constant'
acquisition_function = 'UCB'
corr_kernel = 'exponential_periodic' #'squared_exponential' # 


# nb_features, feat_select,max_ngram_range,max_df, min_df, alpha_NB, mode_tfidf
parameters = { '0' : ['int',[13,32]],
               '1' : ['int',[5,10]],
               '2' : ['int',[1,4]],
               '3' : ['int',[4,10]],
               '4' : ['int',[0,1]],
               '5' : ['int',[1,10]],
               '6' : ['int',[1,3]], }

runExperiment(first_exp = first_exp,
              n_exp = n_exp,
              model = model,
              parameters = parameters,
              n_random_init = n_random_init,
              n_total_iter = n_iter,
              corr_kernel = corr_kernel,
              acquisition_function = acquisition_function,
              n_clusters = n_clusters,
              cluster_evol = cluster_evol,
              GCP_mapWithNoise = GCP_mapWithNoise,
              GCP_useAllNoisyY = GCP_useAllNoisyY,
              model_noise = model_noise,
              n_candidates = n_candidates)