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
n_smart_iter = 15
n_clusters = 1
cluster_evol = 'constant'
acquisition_function = 'MaxUpperBound'
corr_kernel = 'exponential_periodic' #'squared_exponential' # 


# nb_features, feat_select,max_ngram_range,max_df, min_df, alpha_NB, mode_tfidf
parameter_bounds = np.asarray( [
        [13,33],
        [5,11],
        [1,5],
        [4,11],
        [0,2],
        [1,11],
        [1,4]] )


runExperiment(first_exp=first_exp,
              n_exp=n_exp,
              model=model,
              parameter_bounds=parameter_bounds,
              n_random_init=n_random_init,
              n_smart_iter=n_smart_iter,
              corr_kernel = corr_kernel,
              acquisition_function = acquisition_function,
              n_clusters = n_clusters,
              cluster_evol = cluster_evol,
              GCP_mapWithNoise=GCP_mapWithNoise,
              GCP_useAllNoisyY=GCP_useAllNoisyY,
              model_noise=model_noise,
              n_candidates=n_candidates)