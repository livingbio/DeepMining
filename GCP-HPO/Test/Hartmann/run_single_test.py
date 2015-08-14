import numpy as np
import sys
from har6 import har6

sys.path.append("../../")
from smart_search import SmartSearch

### Set parameters ###
parameters = { 'a' : ['float',[0,1]],
			   'b' : ['float',[0,1]],
			   'c' : ['float',[0,1]],
			   'd' : ['float',[0,1]],
			   'e' : ['float',[0,1]],
			   'f' : ['float',[0,1]] }
			   
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
	p_vector = [p_dict['a'],
				p_dict['b'],
				p_dict['c'],
				p_dict['d'],
				p_dict['e'],
				p_dict['f'] ]
	return har6(p_vector)

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