import numpy as np
import sys
from branin import branin


sys.path.append("../../")
from smart_sampling import smartSampling

### Set parameters ###
parameter_bounds = np.asarray( [[0,15],[0,15]] )
nugget = 1.e-10
n_clusters = 1
cluster_evol ='constant'
corr_kernel = 'squared_exponential'
mapWithNoise= False
model_noise = None
sampling_model = 'GCP'
n_candidates= 300
n_random_init= 15
nb_GCP_steps = 85
nb_iter_final = 0
acquisition_function = 'MaxUpperBound'


def scoring_function(p_vector):
	x,y = p_vector
	x = x -5.
	y= y
	return branin(x,y)


X,Y = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,
											  isInt=False,
							                  corr_kernel = corr_kernel,
							                  acquisition_function = acquisition_function,
							                  GCP_mapWithNoise=mapWithNoise,
							          		  model_noise = model_noise,
							                  model = sampling_model, 
							                  n_candidates=n_candidates,
							                  n_random_init=n_random_init,
							                  n_final_iter=nb_iter_final,
							                  n_clusters=n_clusters, 
							                  cluster_evol = cluster_evol,
							                  verbose=True,
							                  detailed_res = False)
