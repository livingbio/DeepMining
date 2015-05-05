# -*- coding: utf-8 -*-

# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT



#------------------------------------ Utilities for smartSampling ------------------------------------#


def find_best_cendidate_with_GCP(model,rand_candidates,verbose,acquisition_function='Simple'):
	
	if(acquisition_function=='Simple'):
	
		predictions = model.predict(rand_candidates,eval_MSE=False,eval_confidence_bounds=False)
		best_candidate_idx = np.argmax(predictions)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx]	
	
	elif(acquisition_function=='MaxEstimatedUpperBound'):
	
		predictions,MSE,coefL,coefU = model.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=False)
		upperBound = predictions + 1.96*coefU*np.sqrt(MSE)
		best_candidate_idx = np.argmax(upperBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], upperBound[best_candidate_idx]
	
	elif(acquisition_function=='MaxUpperBound'):
	
		predictions,MSE,boundL,boundU = model.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=True)
		best_candidate_idx = np.argmax(boundU)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'Hopefully :', best_candidate, predictions[best_candidate_idx], boundU[best_candidate_idx]
	
	else:
		print('Acquisition function not handled...')

	return best_candidate
		

		
def find_best_cendidate_with_GP(model,rand_candidates,verbose,acquisition_function='Simple'):
	
	if(acquisition_function=='Simple'):
	
		predictions = model.predict(rand_candidates,eval_MSE=False)
		best_candidate_idx = np.argmax(predictions)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx]	
	
	elif(acquisition_function=='MaxEstimatedUpperBound' or acquisition_function=='MaxUpperBound'):
	
		predictions,MSE = model.predict(rand_candidates,eval_MSE=True)
		upperBound = predictions + 1.96*np.sqrt(MSE)
		best_candidate_idx = np.argmax(upperBound)
		best_candidate = rand_candidates[best_candidate_idx]
		if(verbose):
			print 'GP Hopefully :', best_candidate, predictions[best_candidate_idx], upperBound[best_candidate_idx]
	
	else:
		print('Acquisition function not handled...')

	return best_candidate
		

		
def sample_random_candidates(nb_parameter_sampling,parameter_bounds,isInt):
	n_parameters = parameter_bounds.shape[0]
	candidates = []
	for k in range(n_parameters):
		if(isInt):
			k_sample  = np.asarray( np.random.rand(nb_parameter_sampling) * np.float(parameter_bounds[k][1]-parameter_bounds[k][0]) + parameter_bounds[k][0] ,
								dtype = np.int32)
		else:
			k_sample  = np.asarray( np.random.rand(nb_parameter_sampling) * np.float(parameter_bounds[k][1]-parameter_bounds[k][0]) + parameter_bounds[k][0] )
		candidates.append(k_sample)
	candidates = np.asarray(candidates)
	candidates = candidates.T
	
	return compute_unique1(candidates)

	
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