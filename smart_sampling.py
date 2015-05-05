# -*- coding: utf-8 -*-

# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT


import numpy as np
from gcp_v2 import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from random import randint, randrange


def smartSampling(nb_iter,
				   parameter_bounds,
				   score_function,
				   model='GCP',
				   acquisition_function='MaxUpperBound',
				   corr_kernel= 'exponential_periodic',
				   nb_random_steps=30,
				   nb_parameter_sampling=2000,
				   n_clusters=1,
				   isInt=True,
				   returnOutputs=False,
				   verbose=False):

	# nb_iter : the number of smart iterations to perform

	# nb_random_steps : the number of random iterations to perform before the smart sampling

	# parameters_bounds : the bounds between which to sample the parameters 
	#		parameter_bounds.shape = [n_parameters,2]
	#		parameter_bounds[i] = [ lower bound for parameter i, upper bound for parameter i]

	# score_function : callable, a function that computes the output, given some parameters

	# model (string) : the model to run. Choose between :
	#		- GCP (runs only the Gaussian Copula Process)
	#		- GP (runs only the Gaussian Process)
	#		- both (runs both models)
	
	# acquisition function (string) : the function to maximize
	# 		- Simple : maximize the predicted output
	#		- MaxEstimatedUpperBound : maximize the (linear estimation of) the 95% confidence upper bound
	#		- MaxUpperBound : maximize the 95% confidence upper bound

	# corr_kernel (string) : the correlation kernel to choose for the GCP. 
	#		Possible choices are :
	#		- exponential_periodic (a linear combination of 3 classic kernels)
	#		- squared_exponential
	
	# nb_parameter_sampling : the number of random parameters to consider for each GCP / GP iterations
	
	# n_clusters : number of clusters used in the parameter space to build a variable mapping for the GCP
	# 		Note : n_clusters should stay quite small, and nb_random_steps should be >> n_clusters
	
	# isInt : if True, all parameters are considered to be integers
	#		It is better to fix isInt=True rather than converting floating parameters as integers in the scoring
	# 		function, because this would generate a discontinuous scoring function (whereas GP / GCP assume that
	#		the function is smooth)
	
	
	
	# returns :
	# 	- if model == GCP or GP
	#		best_parameter = the parameters that provided the best performances with regards to the scoring function
	#	- if model == both
	#		best_parameter, best_parameter_GP = best parameters according to the GCP / GP
	
	
	
	#----------------- ToDo : ------------------------------------------------------#
	# - enable the user to choose int or float parameters
	# - add the possibility to choose the centroids

	n_parameters = parameter_bounds.shape[0]
	nugget = 0.00001/1000.
	nb_iter_final = 5 ## final steps to search the max
	
	runGCP = False
	runGP = False
	if(model == 'both'):
		runGCP = True
		runGP = True
	elif(model == 'GCP'):
		runGCP = True
	elif(model == 'GP'):
		runGP = True
		
	# to store the results
	parameters = None
	outputs = None


	#-------------------- Random initialization --------------------#

	for i in range(nb_random_steps):
		rand_candidate = np.zeros(n_parameters)
		for j in range(n_parameters):
			if(isInt):
				rand_candidate[j] = randint(parameter_bounds[j][0],parameter_bounds[j][1])
			else:
				rand_candidate[j] = randrange(parameter_bounds[j][0],parameter_bounds[j][1])
		if(isInt):
			rand_candidate = np.asarray( rand_candidate, dtype=np.int32)
		output = score_function(rand_candidate)
		
		if(verbose):
			print('Random try '+str(rand_candidate)+', score : '+str(output))
			
		if(parameters is None):
			parameters = np.asarray([rand_candidate])
			outputs = np.asarray([output])
		else:	
			parameters = np.concatenate((parameters,[rand_candidate]))
			outputs = np.concatenate((outputs,[output]))
		
		parameters,outputs = compute_unique2(parameters,outputs)

	if(runGP):
		parameters_GP = np.copy(parameters)
		outputs_GP = np.copy(outputs)
		
		
	#------------------------ Smart Sampling ------------------------#

	for i in range(nb_iter):

		if(verbose):
			print('Step '+str(i))

		rand_candidates = sample_random_candidates(nb_parameter_sampling,parameter_bounds,isInt)
		if(verbose):
			print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
		
		if(runGCP):
			gcp = GaussianCopulaProcess(nugget = nugget,
										corr=corr_kernel,
										random_start=5,
										n_clusters=n_clusters,
										try_optimize=True)
			gcp.fit(parameters,outputs)
			if verbose:
				print ('GCP theta :'+str(gcp.theta))
				
			best_candidate = find_best_cendidate_with_GCP(gcp,
														  rand_candidates,
														  verbose,
														  acquisition_function)
			output = score_function(best_candidate)
			
			if(verbose):
				print 'Test paramter:', best_candidate,' - ***** accuracy:',output
			
			parameters = np.concatenate((parameters,[best_candidate]))
			outputs = np.concatenate((outputs,[output]))
			
			parameters,outputs = compute_unique2(parameters,outputs)

		# do the same with a GP if asked, to compare the results
		if(runGP):
			gp = GaussianProcess(theta0=1. ,
								 thetaL = 0.001,
								 thetaU = 10.,
								 nugget=nugget)
			gp.fit(parameters_GP,outputs_GP)
			if verbose:
				print ('GP theta :'+str(gp.theta_))
			best_candidate_GP = find_best_cendidate_with_GP(gp,
															rand_candidates,
															verbose,
															acquisition_function)
			output_GP = score_function(best_candidate_GP)
			if(verbose):
				print 'GP Test paramter:', best_candidate_GP,' - ***** accuracy:',output_GP
			parameters_GP = np.concatenate((parameters_GP,[best_candidate_GP]))
			outputs_GP = np.concatenate((outputs_GP,[output_GP]))
			parameters_GP,outputs_GP = compute_unique2(parameters_GP,outputs_GP)


	#----------------- Last step : Try to find the max -----------------#

	if(verbose):
		print('\n*** Last step : try to find the best parameters ***')
	
	for i in range(nb_iter_final):

		if(verbose):
			print('Final step '+str(i))
			
		if(runGCP):
			rand_candidates = sample_random_candidates(4*nb_parameter_sampling,parameter_bounds,isInt)
			if(verbose):
				print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
			
			gcp = GaussianCopulaProcess(nugget = nugget,
										random_start=5,
										n_clusters=n_clusters,
										try_optimize=True)
			gcp.fit(parameters,outputs)
			if verbose:
				print ('GCP theta :'+str(gcp.theta))
				
			best_candidate = find_best_cendidate_with_GCP(gcp,
														  rand_candidates,
														  verbose)
			output = score_function(best_candidate)
			
			if(verbose):
				print 'Test paramter:', best_candidate,' - accuracy:',output
			
			parameters = np.concatenate((parameters,[best_candidate]))
			outputs = np.concatenate((outputs,[output]))
			parameters,outputs = compute_unique2(parameters,outputs)

		# do the same with a GP if asked, to compare the results
		if(runGP):
			gp = GaussianProcess(theta0=1. ,
								 thetaL = 0.001,
								 thetaU = 10.,
								 nugget=nugget)
			gp.fit(parameters_GP,outputs_GP)
			best_candidate_GP = find_best_cendidate_with_GP(gp,
															rand_candidates,
															verbose)
			output_GP = score_function(best_candidate_GP)
			if(verbose):
				print 'GP Test paramter:', best_candidate_GP,' - accuracy:',output_GP
			parameters_GP = np.concatenate((parameters_GP,[best_candidate_GP]))
			outputs_GP = np.concatenate((outputs_GP,[output_GP]))
			parameters_GP,outputs_GP = compute_unique2(parameters_GP,outputs_GP)
			
			
	#--------------------------- Final Result ---------------------------#

	if(runGCP):
		best_parameter = np.argmax(outputs)
		print('Best parameters '+str(parameters[best_parameter]) + ' with output: ' + str(outputs[best_parameter]))

	if(runGP):
		best_parameter_GP = np.argmax(outputs_GP)
		print('GP Best parameters '+str(parameters_GP[best_parameter_GP]) + ' with output: ' + str(outputs_GP[best_parameter_GP]))
	
	if(model == 'both'):
		if(returnOutputs):
			return best_parameter, outputs, outputs_GP
		else:
			return best_parameter,best_parameter_GP
	
	elif(model == 'GCP'):
		return best_parameter
	
	elif(model == 'GP'):
		return best_parameter_GP
	

	

	
#------------------------------------ Utilities ------------------------------------#


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