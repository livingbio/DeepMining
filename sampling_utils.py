# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

import numpy as np
from random import randint, randrange
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from scipy import integrate


max_f_value = 1.

#------------------------------------ Utilities for smartSampling ------------------------------------#

def find_best_candidate(model, X, raw_Y,mean_Y,std_Y, data_size_bounds,args, rand_candidates,verbose,acquisition_function='Simple'):
	
	if(model == 0):
		best_candidate = find_best_candidate_with_GCP(X, raw_Y, mean_Y, std_Y, data_size_bounds,args, rand_candidates,verbose,acquisition_function)
		
	elif(model == 1):
		best_candidate = find_best_candidate_with_GP(X, mean_Y, data_size_bounds, args, rand_candidates,verbose,acquisition_function)
		
	elif(model == 2):
		best_candidate = rand_candidates[ randint(0,rand_candidates.shape[0]-1)]
		
	else:
		print('Error in find_best_candidate')
		
	return best_candidate

	
def find_best_candidate_with_GCP(X, raw_Y, mean_Y, std_Y, data_size_bounds, args, rand_candidates,verbose,acquisition_function='Simple'):
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

	if verbose:
		print ('GCP theta :'+str(mean_gcp.theta))
				
	if(acquisition_function=='Simple'):
	
		predictions = mean_gcp.predict(rand_candidates,eval_MSE=False,eval_confidence_bounds=False)
		best_candidate_idx = np.argmax(predictions)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx]	
	
	elif(acquisition_function=='MaxEstimatedUpperBound'):
	
		predictions,MSE,coefL,coefU = mean_gcp.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=False)
		upperBound = predictions + 1.96*coefU*np.sqrt(MSE)
		best_candidate_idx = np.argmax(upperBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], upperBound[best_candidate_idx]
	
	elif(acquisition_function=='MaxUpperBound'):
	
		predictions,MSE,boundL,boundU = \
				mean_gcp.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=True,upperBoundCoef=GCP_upperBound_coef)
		best_candidate_idx = np.argmax(boundU)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], boundU[best_candidate_idx]

	elif(acquisition_function=='MaxLowerBound'):
	
		predictions,MSE,boundL,boundU = \
				mean_gcp.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=True,upperBoundCoef=GCP_upperBound_coef)
		best_candidate_idx = np.argmax(boundL)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], boundL[best_candidate_idx],boundU[best_candidate_idx]

	elif(acquisition_function=='EI'):
	
		predictions,MSE = \
				mean_gcp.predict(rand_candidates,eval_MSE=True,transformY=False) # we want the predictions in the GP space
		y_best = np.max(mean_Y)
		sigma = np.sqrt(MSE)
		ei = [ compute_ei((rand_candidates[i]-mean_gcp.X_mean)/mean_gcp.X_std,predictions[i],sigma[i],y_best, \
						mean_gcp.mapping,mean_gcp.mapping_derivate) \
				for i in range(rand_candidates.shape[0]) ]

		best_candidate_idx = np.argmax(ei)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], ei[best_candidate_idx]

	else:
		print('Acquisition function not handled...')

	return best_candidate
		

		
def find_best_candidate_with_GP(X, Y, data_size_bounds, args, rand_candidates,verbose,acquisition_function='Simple'):
	nugget = args[5]

	gp = GaussianProcess(theta0=1. ,
						 thetaL = 0.001,
						 thetaU = 10.,
						 nugget=nugget)
	gp.fit(X,Y)
	if verbose:
		print ('GP theta :'+str(gp.theta_))
			
	if(acquisition_function=='Simple'):
	
		predictions = gp.predict(rand_candidates,eval_MSE=False)
		best_candidate_idx = np.argmax(predictions)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx]	
	
	elif(acquisition_function=='MaxEstimatedUpperBound' or acquisition_function=='MaxUpperBound'):
	
		predictions,MSE = gp.predict(rand_candidates,eval_MSE=True)
		upperBound = predictions + 1.96*np.sqrt(MSE)
		best_candidate_idx = np.argmax(upperBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx], upperBound[best_candidate_idx]

	elif(acquisition_function=='MaxLowerBound'):
	
		predictions,MSE = gp.predict(rand_candidates,eval_MSE=True)
		lowerBound = predictions - 1.96*np.sqrt(MSE)
		best_candidate_idx = np.argmax(lowerBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx], lowerBound[best_candidate_idx]
	
	else:
		print('Acquisition function not handled...')

	return best_candidate
		

		
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


def compute_ei(x,m,sigma,f_best,Psi,Psi_prim):
	if(f_best > max_f_value):
		print('Error in compute_ei : f_best > max_f_value ')
	def f_to_integrate(u):
		temp = u * Psi_prim(x,f_best+u,normalize=True) / sigma
		temp = temp * np.exp( - 0.5 * ((m - Psi(x,f_best+u,normalize=True)[0])/ sigma )**2. )
		return temp
	return integrate.quad(f_to_integrate,0,(2.*max_f_value)-f_best)[0]


def compute_unique1(a):
	#http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

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
