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
				   data_size_bounds = None,
				   model='GCP',
				   acquisition_function='MaxUpperBound',
				   corr_kernel= 'exponential_periodic',
				   nb_random_steps=30,
				   nb_iter_final = 5,
				   nb_parameter_sampling=2000,
				   n_clusters=1,
				   cluster_evol = 'constant',
   				   GCPconsiderAllObs1=True,
				   GCPconsiderAllObs2=True,
				   isInt=True,
				   returnAllParameters=True,
				   verbose=False):

	# nb_iter : the number of smart iterations to perform

	# nb_random_steps : the number of random iterations to perform before the smart sampling

	# parameters_bounds : the bounds between which to sample the parameters 
	#		parameter_bounds.shape = [n_parameters,2]
	#		parameter_bounds[i] = [ lower bound for parameter i, upper bound for parameter i]
	
	# score_function : callable, a function that computes the output, given some parameters
	#       /!\ Always put data_size as the first parameter

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
	if(cluster_evol != 'constant'):
		GCP_args = [corr_kernel, 1,GCPconsiderAllObs1,GCPconsiderAllObs2]
	else:
		GCP_args = [corr_kernel, n_clusters,GCPconsiderAllObs1,GCPconsiderAllObs2]
	GCP_args_with_clusers = [corr_kernel, n_clusters,GCPconsiderAllObs1,GCPconsiderAllObs2]
			
	if(verbose):
		print 'parameter bounds :',parameter_bounds
		print 'n_parameters :', n_parameters
		print 'Nbr of final steps :', nb_iter_final
		print 'GCP args :',GCP_args
		print 'Data size can vary between',data_size_bounds
		print_utils_parameters()
	
	### models' order : GCP, GP, random
	nb_model = 3
	modelToRun = np.zeros(nb_model)
	if(model == 'all'):
		modelToRun = np.asarray([1,1,1])
	elif(model == 'GCPR'):
		modelToRun = np.asarray([1,0,1])	
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
	raw_outputs = None


	#-------------------- Random initialization --------------------#

	# sample nb_random_steps random parameters to initialize the process
	init_rand_candidates = sample_random_candidates_for_init(nb_random_steps,parameter_bounds,data_size_bounds,isInt)
	for i in range(init_rand_candidates.shape[0]):
		print i
		rand_candidate = init_rand_candidates[i]
		new_output = score_function(rand_candidate)
		
		if(verbose):
			print('Random try '+str(rand_candidate)+', score : '+str(np.mean(new_output)))
			
		if(parameters is None):
			parameters = np.asarray([rand_candidate])
			raw_outputs = [new_output]
			mean_outputs = [np.mean(new_output)]
			std_outputs = [np.std(new_output)]
		else:
			parameters,raw_outputs,mean_outputs,std_outputs = \
				add_results(parameters,raw_outputs,mean_outputs,std_outputs,rand_candidate,new_output)		

	all_parameters = []
	all_raw_outputs = []
	all_mean_outputs = []
	all_std_outputs = []
	for i in range(nb_model):
		if(modelToRun[i]):
			all_parameters.append(np.copy(parameters))
			all_raw_outputs.append(list(raw_outputs))
			all_mean_outputs.append(list(mean_outputs))
			all_std_outputs.append(list(std_outputs))
	# all_parameters = np.asarray(all_parameters)
	# all_raw_outputs = np.asarray( all_raw_outputs)
	print(all_parameters[0].shape)
		
		
	#------------------------ Smart Sampling ------------------------#
	
	i_mod_10 = 0

	for i in range(nb_iter):
		if(i==20 and cluster_evol=='step'):
			GCP_args = GCP_args_with_clusters

		if(i/10 > (i_mod_10+1) and cluster_evol=='variable'):
			GCP_args[0] = GCP_args[0]
			GCP_args[1] = GCP_args[1]+1
			i_mod_10 += 2

		if(verbose):
			print('Step '+str(i))
			model_idx=0
			for k in range(nb_model):
				if(modelToRun[k]):
					print k,'current best output',np.max(all_mean_outputs[model_idx])
					model_idx += 1	
				
		rand_candidates = sample_random_candidates(nb_parameter_sampling,parameter_bounds,data_size_bounds,isInt)
		
		if(verbose):
			print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
		
		model_idx = 0
		for k in range(nb_model):
			if(modelToRun[k]):
				best_candidate = find_best_candidate(k,
													 all_parameters[model_idx],
													 all_raw_outputs[model_idx],
													 all_mean_outputs[model_idx],
													 all_std_outputs[model_idx],
													 data_size_bounds,
													 GCP_args,
													 rand_candidates,
													 verbose,
													 acquisition_function)
				new_output = score_function(best_candidate)

				all_parameters[model_idx],all_raw_outputs[model_idx],\
					all_mean_outputs[model_idx],all_std_outputs[model_idx] = \
						add_results(all_parameters[model_idx],all_raw_outputs[model_idx],\
							all_mean_outputs[model_idx],all_std_outputs[model_idx],\
								best_candidate,new_output)		

				model_idx += 1
					
				if(verbose):
					print k,'Test paramter:', best_candidate,' - ***** accuracy:',new_output
					#print 'mean outputs'
					# print(all_mean_outputs)
					# print '\n'		

	#----------------- Last step : Try to find the max -----------------#

	if(verbose):
		print('\n*** Last step : try to find the best parameters ***')
		
	if(data_size_bounds is not None):
		data_size_bounds[0] = data_size_bounds[1]
		if(verbose):
			print('Fixed the data size at',data_size_bounds[0])
	
	for i in range(nb_iter_final):

		if(verbose):
			print('Final step '+str(i))
		
		rand_candidates = sample_random_candidates(nb_parameter_sampling,parameter_bounds,data_size_bounds,isInt)
		
		if(verbose):
			print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
		
		model_idx = 0
		for k in range(nb_model):
			if(modelToRun[k]):
				best_candidate = find_best_candidate(k,
													 all_parameters[model_idx],
													 all_raw_outputs[model_idx],
													 all_mean_outputs[model_idx],
													 all_std_outputs[model_idx],
													 data_size_bounds,
													 GCP_args,
													 rand_candidates,
													 verbose,
													 acquisition_function='Simple')
				new_output = score_function(best_candidate)

				all_parameters[model_idx],all_raw_outputs[model_idx],\
					all_mean_outputs[model_idx],all_std_outputs[model_idx] = \
						add_results(all_parameters[model_idx],all_raw_outputs[model_idx],\
							all_mean_outputs[model_idx],all_std_outputs[model_idx],\
								best_candidate,new_output)		

				model_idx += 1
					
				if(verbose):
					print k,'Test paramter:', best_candidate,' - ***** accuracy:',new_output
			

	#--------------------------- Final Result ---------------------------#

	##### ToDo ####
	##### Return something else that the parameters that max mean_outputs (=mean)

	best_parameters = []
	
	model_idx = 0
	for k in range(nb_model):
		if(modelToRun[k]):
			best_parameter_idx = np.argmax(all_mean_outputs[model_idx])
			best_parameters.append(all_parameters[model_idx][best_parameter_idx])
			if(data_size_bounds is not None):
				best_parameter_idx2 = np.argmax(all_mean_outputs[model_idx][all_parameters[model_idx][:,0] == data_size_bounds[1]])
			print k,'Best parameters '+str(all_parameters[model_idx][best_parameter_idx]) + ' with output: ' + str(all_mean_outputs[model_idx][best_parameter_idx])
			if(data_size_bounds is not None):
				print k,'Best parameters for complete dataset'+ \
									str( (all_parameters[model_idx][all_parameters[model_idx][:,0] == data_size_bounds[1]])[best_parameter_idx2]) \
									+ ' with output: ' + \
									str( (all_mean_outputs[model_idx][all_parameters[model_idx][:,0] == data_size_bounds[1]])[best_parameter_idx2])

			model_idx += 1
	best_parameters = np.asarray(best_parameters)
	
	if(verbose):
		print '\n','n_parameters :', n_parameters
		print 'Nbr of final steps :', nb_iter_final
		print 'GCP args :',GCP_args
		print_utils_parameters()
	
	if(returnAllParameters):
		return all_parameters , all_raw_outputs, all_mean_outputs, all_std_outputs
	else:
		return all_parameters, all_raw_outputs

	

	
