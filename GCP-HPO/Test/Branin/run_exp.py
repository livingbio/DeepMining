import numpy as np
import sys
from branin import branin


sys.path.append("../../")
from smart_search import SmartSearch

### Set parameters ###
parameters = { 'x' : ['float',[0,15]],
			   'y' : ['float',[0,15]] }
nugget = 1.e-10
n_clusters = 1
cluster_evol ='constant'
corr_kernel = 'squared_exponential'
mapWithNoise= False
model_noise = None
sampling_model = 'GCP'
n_candidates= 300
n_random_init= 15
n_iter = 100
nb_iter_final = 0
acquisition_function = 'UCB'


def scoring_function(p_dict):
	x,y = p_dict['x'], p_dict['y']
	x = x -5.
	y= y
	return branin(x,y)


search = SmartSearch(parameters,
			estimator=scoring_function,
			corr_kernel = corr_kernel,
			acquisition_function = acquisition_function,
			GCP_mapWithNoise=mapWithNoise,
			model_noise = model_noise,
			model = sampling_model, 
			n_candidates=n_candidates,
			n_iter = n_iter,
			n_init = n_random_init,
			n_final_iter=nb_iter_final,
			n_clusters=n_clusters, 
			cluster_evol = cluster_evol,
			verbose=2,
			detailed_res = 0)

search._fit()

