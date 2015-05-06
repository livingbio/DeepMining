# -*- coding: utf-8 -*-

# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT


import numpy as np
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from random import randint, randrange
from sampling_utils import *

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
	#		- random (samples at random)
	#		- all (runs all models)
	
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
	
	# isInt : bool or (n_parameters) numpy array, specify which parameters are integers
	#		If isInt is a boolean, all parameters are assumed to have the same type.
	#		It is better to fix isInt=True rather than converting floating parameters as integers in the scoring
	# 		function, because this would generate a discontinuous scoring function (whereas GP / GCP assume that
	#		the function is smooth)
	
	
	
	# returns :
	# 	- if model == GCP , GP or random
	#		best_parameter = the parameters that provided the best performances with regards to the scoring function
	#	- if model == all
	#		best_parameter, best_parameter_GP = best parameters according to the GCP / GP
	
	
	
	#----------------- ToDo : ------------------------------------------------------#
	# - add the possibility to choose the centroids
	# - add non-isotropic models

	
	#---------------------------- Init ----------------------------#
	n_parameters = parameter_bounds.shape[0]
	nb_iter_final = 3 ## final steps to search the max
	GCP_args = [corr_kernel, n_clusters]
	
	### models' order : GCP, GP, random
	nb_model = 3
	modelToRun = np.zeros(nb_model)
	if(model == 'all'):
		modelToRun = np.asarray([1,1,1])
	elif(model == 'GCP'):
		modelToRun[0] = 1
	elif(model == 'GP'):
		modelToRun[1] = 1
	elif(model == 'random'):
		modelToRun[2] = 1
		
	# transform isInt into a (n_parameters) numpy array
	if not(type(isInt).__name__ == 'ndarray'):
		b= isInt
		if(b):
			isInt = np.ones(n_parameters)
		else:
			isInt = np.zeros(n_parameters)
	else:
		if not (isInt.shape[0] == n_parameters):
			print 'Warning : isInt array has not the right shape'
	
	# to store the results
	parameters = None
	outputs = None


	#-------------------- Random initialization --------------------#

	# sample nb_random_steps random parameters to initialize the process
	init_rand_candidates = sample_random_candidates(nb_random_steps,parameter_bounds,isInt)
	for i in range(init_rand_candidates.shape[0]):
		rand_candidate = init_rand_candidates[i]
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

	all_parameters = []
	all_outputs = []
	for i in range(nb_model):
		if(modelToRun[i]):
			all_parameters.append(np.copy(parameters))
			all_outputs.append(outputs)
	all_parameters = np.asarray(all_parameters)
	all_outputs = np.asarray( all_outputs)
	print(all_parameters.shape)
		
		
	#------------------------ Smart Sampling ------------------------#

	for i in range(nb_iter):

		if(verbose):
			print('Step '+str(i))

		rand_candidates = sample_random_candidates(nb_parameter_sampling,parameter_bounds,isInt)
		
		if(verbose):
			print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
		
		all_new_parameters = []
		all_new_outputs = []
		model_idx = 0
		for k in range(nb_model):
			if(modelToRun[k]):
				best_candidate = find_best_candidate(k,
													 all_parameters[model_idx],
													 all_outputs[model_idx],
													 GCP_args,
													 rand_candidates,
													 verbose,
													 acquisition_function)
				output = score_function(best_candidate)

				# erase duplicates as it can cause errors in GCP.fit and GCP.predict
				new_parameters,new_outputs = compute_unique2( np.concatenate((all_parameters[model_idx],[best_candidate])),
															  np.concatenate((all_outputs[model_idx],[output])) )
				all_new_parameters.append(new_parameters)
				all_new_outputs.append(new_outputs)

				model_idx += 1
					
				if(verbose):
					print k,'Test paramter:', best_candidate,' - ***** accuracy:',output
			
		# ToDo clean this
		all_parameters = np.asarray(all_new_parameters)
		all_outputs = np.asarray(all_new_outputs)	

				

	#----------------- Last step : Try to find the max -----------------#

	if(verbose):
		print('\n*** Last step : try to find the best parameters ***')
	
	for i in range(nb_iter_final):

		if(verbose):
			print('Final step '+str(i))
		
		rand_candidates = sample_random_candidates(nb_parameter_sampling,parameter_bounds,isInt)
		
		all_new_parameters = []
		all_new_outputs = []
		model_idx = 0
		for k in range(nb_model):
			if(modelToRun[k]):
				# Here we choose acquisition_function == 'HighScoreHighConfidence'
				# to trade-off the high score and high confidence desired
				best_candidate = find_best_candidate(k,
													 all_parameters[model_idx],
													 all_outputs[model_idx],
													 GCP_args,
													 rand_candidates,
													 verbose,
													 acquisition_function='HighScoreHighConfidence')
				output = score_function(best_candidate)

				# erase duplicates as it can cause errors in GCP.fit and GCP.predict
				new_parameters,new_outputs = compute_unique2( np.concatenate((all_parameters[model_idx],[best_candidate])),
															  np.concatenate((all_outputs[model_idx],[output])) )
				all_new_parameters.append(new_parameters)
				all_new_outputs.append(new_outputs)

				model_idx += 1
					
				if(verbose):
					print k,'Test paramter:', best_candidate,' - ***** accuracy:',output
			
		# ToDo clean this
		all_parameters = np.asarray(all_new_parameters)
		all_outputs = np.asarray(all_new_outputs)	
			
	#--------------------------- Final Result ---------------------------#

	best_parameters = []
	
	model_idx = 0
	for k in range(nb_model):
		if(modelToRun[k]):
			best_parameter_idx = np.argmax(all_outputs[model_idx])
			best_parameters.append(all_parameters[model_idx][best_parameter_idx])
			print k,'Best parameters '+str(all_parameters[model_idx][best_parameter_idx]) + ' with output: ' + str(all_outputs[model_idx][best_parameter_idx])
			model_idx += 1
	best_parameters = np.asarray(best_parameters)
	
	if(returnOutputs):
		return best_parameters , all_outputs
	else:
		return best_parameters
	

	

	
