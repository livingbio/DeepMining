# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

# The MIT License (MIT)
# Copyright (c) 2015 Sebastien Dubois

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from random import randint, randrange
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from scipy import integrate
from scipy.stats import norm

max_f_value = 1.

#------------------------------------ Utilities for smartSampling ------------------------------------#

def find_best_candidate(model, X, raw_Y, args, rand_candidates,verbose,acquisition_function='Simple'):
	
	mean_Y,std_Y = [],[]
	for o in raw_Y:
		mean_Y.append(np.mean(o))
		std_Y.append(np.std(o))
	mean_Y = np.asarray(mean_Y)
	std_Y = np.asarray(std_Y)

	if(model == 'GCP'):
		best_candidate = find_best_candidate_with_GCP(X, raw_Y, mean_Y, std_Y, args, rand_candidates,verbose,acquisition_function)
		
	elif(model == 'GP'):
		best_candidate = find_best_candidate_with_GP(X, mean_Y, args, rand_candidates,verbose,acquisition_function)
		
	elif(model == 'rand'):
		best_candidate = rand_candidates[ randint(0,rand_candidates.shape[0]-1)]
		
	else:
		print('Error in find_best_candidate')
		
	return best_candidate

	
def find_best_candidate_with_GCP(X, raw_Y, mean_Y, std_Y, args, rand_candidates,verbose,acquisition_function='Simple'):
	corr_kernel = args[0]
	n_clusters = args[1]
	GCP_mapWithNoise = args[2]
	GCP_useAllNoisyY = args[3]
	GCP_model_noise = args[4]
	nugget = args[5]
	GCP_upperBound_coef = args[6]

	mean_gcp = GaussianCopulaProcess(nugget = nugget,
									corr = corr_kernel,
									random_start = 5,
									n_clusters = n_clusters,
								 	mapWithNoise = GCP_mapWithNoise,
					 				useAllNoisyY = GCP_useAllNoisyY,
					 				model_noise = GCP_model_noise,
									try_optimize = True)
	mean_gcp.fit(X,mean_Y,raw_Y,obs_noise=std_Y)

	if(verbose == 2):
		print ('GCP theta :'+str(mean_gcp.theta))
				
	if(acquisition_function=='Simple'):
	
		predictions = mean_gcp.predict(rand_candidates,eval_MSE=False,eval_confidence_bounds=False)
		best_candidate_idx = np.argmax(predictions)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx]	
	
	elif(acquisition_function=='UCB'):
	
		predictions,MSE,boundL,boundU = \
				mean_gcp.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=True,coef_bound = GCP_upperBound_coef)
		best_candidate_idx = np.argmax(boundU)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], boundU[best_candidate_idx]

	elif(acquisition_function=='MaxLowerBound'):
	
		predictions,MSE,boundL,boundU = \
				mean_gcp.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=True,coef_bound = GCP_upperBound_coef)
		best_candidate_idx = np.argmax(boundL)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], boundL[best_candidate_idx],boundU[best_candidate_idx]

	elif(acquisition_function=='EI'):
	
		predictions,MSE = \
				mean_gcp.predict(rand_candidates,eval_MSE=True,transformY=False) # we want the predictions in the GP space
		y_best = np.max(mean_Y)
		sigma = np.sqrt(MSE)
		ei = [ gcp_compute_ei((rand_candidates[i]-mean_gcp.X_mean)/mean_gcp.X_std,predictions[i],sigma[i],y_best, \
						mean_gcp.mapping,mean_gcp.mapping_derivative) \
				for i in range(rand_candidates.shape[0]) ]

		best_candidate_idx = np.argmax(ei)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], ei[best_candidate_idx]

	else:
		print('Acquisition function not handled...')

	return best_candidate
		

		
def find_best_candidate_with_GP(X, Y, args, rand_candidates,verbose,acquisition_function='Simple'):
	nugget = args[5]

	gp = GaussianProcess(theta0=1. * np.ones(X.shape[1]) ,
						 thetaL = 0.001 * np.ones(X.shape[1]) ,
						 thetaU = 10. * np.ones(X.shape[1]) ,
						 nugget=nugget)
	gp.fit(X,Y)
	if(verbose == 2):
		print ('GP theta :'+str(gp.theta_))
			
	if(acquisition_function=='Simple'):
	
		predictions = gp.predict(rand_candidates,eval_MSE=False)
		best_candidate_idx = np.argmax(predictions)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx]	
	
	elif(acquisition_function=='UCB'):
	
		predictions,MSE = gp.predict(rand_candidates,eval_MSE=True)
		upperBound = predictions + 1.96*np.sqrt(MSE)
		best_candidate_idx = np.argmax(upperBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx], upperBound[best_candidate_idx]

	elif(acquisition_function=='EI'):
	
		predictions,MSE = gp.predict(rand_candidates,eval_MSE=True)
		y_best = np.max(Y)
		sigma = np.sqrt(MSE)
		ei = [ gp_compute_ei(predictions[i],sigma[i],y_best) \
				for i in range(rand_candidates.shape[0]) ]
		best_candidate_idx = np.argmax(ei)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx], upperBound[best_candidate_idx]

	elif(acquisition_function=='MaxLowerBound'):
	
		predictions,MSE = gp.predict(rand_candidates,eval_MSE=True)
		lowerBound = predictions - 1.96*np.sqrt(MSE)
		best_candidate_idx = np.argmax(lowerBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose == 2):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx], lowerBound[best_candidate_idx]
	
	else:
		print('Acquisition function not handled...')

	return best_candidate
		
def sample_candidates(n_candidates,param_bounds,param_isInt):
	n_parameters = param_isInt.shape[0]
	candidates = []

	for k in range(n_parameters):
		if(param_isInt[k]):
			k_sample  = np.asarray( np.random.rand(n_candidates) * np.float(param_bounds[k][1]-param_bounds[k][0]) + param_bounds[k][0] ,
								dtype = np.int32)
		else:
			k_sample  = np.asarray( np.random.rand(n_candidates) * np.float(param_bounds[k][1]-param_bounds[k][0]) + param_bounds[k][0] )
		candidates.append(k_sample)

	candidates = np.asarray(candidates)
	candidates = candidates.T

	return compute_unique1(candidates)
		
def sample_random_candidates(n_candidates,parameter_bounds,data_size_bounds,isInt):
	n_parameters = isInt.shape[0]
	candidates = []
	if(data_size_bounds is not None):
		# favor small data sizes
		# data_size_samples = np.asarray( (data_size_bounds[0] + (1-np.sqrt(np.random.rand(1)))*(data_size_bounds[1]-data_size_bounds[0]))
		#								* np.ones(n_candidates),
		#								dtype = np.int32 )
		
		# sample data size uniformly
		data_size_samples = np.asarray( (data_size_bounds[0] + (np.random.rand(1))*(data_size_bounds[1]-data_size_bounds[0]))
										* np.ones(n_candidates),
										dtype = np.int32 )
		candidates.append(data_size_samples)

	for k in range(n_parameters):
		if(isInt[k]):
			k_sample  = np.asarray( np.random.rand(n_candidates) * np.float(parameter_bounds[k][1]-parameter_bounds[k][0]) + parameter_bounds[k][0] ,
								dtype = np.int32)
		else:
			k_sample  = np.asarray( np.random.rand(n_candidates) * np.float(parameter_bounds[k][1]-parameter_bounds[k][0]) + parameter_bounds[k][0] )
		candidates.append(k_sample)

	candidates = np.asarray(candidates)
	candidates = candidates.T
	
	return compute_unique1(candidates)

def sample_random_candidates_for_init(n_candidates,parameter_bounds,data_size_bounds,isInt):
	n_parameters = isInt.shape[0]
	candidates = []
	if(data_size_bounds is not None):
		data_size_samples = np.asarray( (data_size_bounds[0] + (1-np.random.rand(n_candidates))*(data_size_bounds[1]-data_size_bounds[0])),
										dtype = np.int32 )
		candidates.append(data_size_samples)

	for k in range(n_parameters):
		if(isInt[k]):
			k_sample  = np.asarray( np.random.rand(n_candidates) * np.float(parameter_bounds[k][1]-parameter_bounds[k][0]) + parameter_bounds[k][0] ,
								dtype = np.int32)
		else:
			k_sample  = np.asarray( np.random.rand(n_candidates) * np.float(parameter_bounds[k][1]-parameter_bounds[k][0]) + parameter_bounds[k][0] )
		candidates.append(k_sample)
	
	candidates = np.asarray(candidates)
	candidates = candidates.T
	
	return candidates

def add_results(parameters,raw_outputs,score_outputs,std_outputs,new_param,new_output):
	# add a new observation (ie a value returned by the scoring function, so usually a 
	# list of all 5-fold CV results). The new observation should be concatenated to previous
	# ones if the parameters had already been tested, or otherwise just added to the list 
	# of observations

	is_in,idx = is_in_ndarray(new_param,parameters)
	if(is_in):
		# parameters is already in our log
		raw_outputs[idx] += new_output
		# update mean and std for this parameter set
		score_outputs[idx] = np.mean(raw_outputs[idx])
		std_outputs[idx] = np.std(raw_outputs[idx])
	else:
		# this is the first result for this parameter set
		parameters = np.concatenate((parameters,[new_param]))
		raw_outputs.append(new_output)
		score_outputs.append(np.mean(new_output))
		std_outputs.append(np.std(new_output))
	
	return parameters,raw_outputs,score_outputs,std_outputs


def gcp_compute_ei(x,m,sigma,f_best,Psi,Psi_prim):
	# Compute Expected improvement for GCP
	# m,sigma == mean, std from GP predictions
	# f_best == current best value observed 
	# Psi == mapping function
	# Psi_prim == the derivative of the mapping function
	# Note : max_f_value is a boundary to fasten the integration,
	# for example for prediction accuracy ut shouldn't be greater
	# than 1.
	# When calling Psi and Psi_prim, normalize is set to True as here
	# we consider directly the observed values and not the normalized 
	# ones (but usually when fitting the GCP there is a normalization step)

	if(f_best > max_f_value):
		print('Error in compute_ei : f_best > max_f_value ')
	def f_to_integrate(u):
		temp = u * Psi_prim(x,f_best+u,normalize=True) / sigma
		temp = temp * np.exp( - 0.5 * ((m - Psi(x,f_best+u,normalize=True)[0])/ sigma )**2. )
		return temp
	return integrate.quad(f_to_integrate,0,(2.*max_f_value)-f_best)[0]

def gp_ompute_ei(m,sigma,y_best):
	# Compute Expected improvement for GP
	# m,sigma == mean, std from GP predictions
	# f_best == current best value observed 	ei_array = np.zeros(predictions.shape[0])

	z = (y_best - m) / sigma
	ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
	return ei

def compute_unique1(a):
	# keep only unique values in the ndarray a
	# http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

	b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
	_, idx = np.unique(b, return_index=True)
	idx =np.sort(idx)
	return a[idx]
	
	
def compute_unique2(a1,a2):
	# keep only unique rows of a1, and delete the corresponding rows in a2
	
	b = np.ascontiguousarray(a1).view(np.dtype((np.void, a1.dtype.itemsize * a1.shape[1])))
	_, idx = np.unique(b, return_index=True)
	idx =np.sort(idx)
	return a1[idx],a2[idx]	

def is_in_2darray(item,a):
	# look for element item in 2darray a
	# returns True if item is in a, and its index

	idx0 =  a[:,0]==item[0]
	if np.sum(idx0 > 0):
		idx1 = (a[idx0,1] == item[1])
		if(np.sum(idx1) > 0):
			all_idx = np.asarray(range(a.shape[0]))
			return True,((all_idx[idx0])[idx1])[0]
		else:
			return False,0
	else:
		return False,0

def is_in_ndarray(item,a):
	# look for element item in ndarray a
	# returns True if item is in a, and its index
	
	k = 0
	idx_val = np.asarray(range(a.shape[0]))
	idxk = range(a.shape[0])
	while( k < a.shape[1]):
		idxk =  (a[idxk,k]==item[k])
		if(np.sum(idxk > 0)):
			k += 1
			idx_val = idx_val[idxk]
			idxk = list(idx_val) 
		else:
			return False,0

	return True,idx_val[0]
