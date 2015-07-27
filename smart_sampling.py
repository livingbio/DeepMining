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
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from random import randint, randrange
import sampling_utils as utils 

def smartSampling(n_iter,
				   parameter_bounds,
				   score_function,
				   data_size_bounds = None,
				   model='GCP',
				   acquisition_function='MaxUpperBound',
				   corr_kernel= 'exponential_periodic',
				   n_random_init=30,
				   n_final_iter = 5,
				   n_candidates=1000,
				   n_clusters=1,
				   n_clusters_max=5,
				   cluster_evol = 'constant',
				   nugget = 1.e-7,
   				   GCP_mapWithNoise=False,
				   GCP_useAllNoisyY=False,
				   model_noise=None,
				   isInt=True,
				   detailed_res=False,
				   verbose=False):
	"""
	n_iter : int
		Number of smart iterations to perform.

	parameters_bounds : ndarray
		The bounds between which to sample the parameters.
		parameter_bounds.shape = [n_parameters,2]
		parameter_bounds[i] = [ lower bound for parameter i, upper bound for parameter i]
	
	score_function : callable
		A function that computes the output, given some parameters.
		This is the function to optimize.
		/!\ Always put data_size as the first parameter, if not None

	model : string, optional
		The model to run.
		Choose between :
		- GCP (runs only the Gaussian Copula Process)
		- GP (runs only the Gaussian Process)
		- random (samples at random)
		- GCPR (GCP and random)
		- all (runs all models)
	
	acquisition function : string, optional
		Function to maximize in order to choose the next parameter to test.
		- Simple : maximize the predicted output
		- MaxUpperBound : maximize the upper confidence bound
		- MaxLowerBound : maximize the lower confidence bound
		- EI : maximizes the expected improvement
		/!\ EI is not available for GP
		Default is 'MaxUpperBound'

	corr_kernel : string, optional
		Correlation kernel to choose for the GCP. 
		Possible choices are :
		- exponential_periodic (a linear combination of 3 classic kernels)
		- squared_exponential
		Default is 'exponential_periodic'.

	n_random_init : int, optional
		Number of random iterations to perform before the smart sampling.
		Default is 30.

	n_candidates : int, optional
		Number of random candidates to sample for each GCP / GP iterations
		Default is 2000.

	n_clusters : int, optional
		Number of clusters used in the parameter space to build a variable mapping for the GCP.
		Default is 1.
	
	cluster_evol : string {'constant', 'step', 'variable'}, optional
		Method used to set the number of clusters.
		If 'constant', the number of clusters is set with n_clusters.
		If 'step', start with one cluster, and set n_clusters after 20 smart steps.
		If 'variable', start with one cluster and increase n_clusters by one every 30 smart steps.
		Default is constant.

	n_clusters_max : int, optional
		The maximum value for n_clusters.
		Default is 5.

	nugget : float, optional
		The nugget to set for the Gaussian Copula Process or Gaussian Process.
		Default is 1.e-7.

	GCP_mapWithNoise : boolean, optional
		If True and if Y outputs contain multiple noisy observations for the same
		x inputs, then all the noisy observations are used to compute Y's distribution
		and learn the mapping function.
		Otherwise, only the mean of the outputs, for a given input x, is considered.
		Default is False.

	GCP_useAllNoisyY : boolean, optional
		If True and if Y outputs contain multiple noisy observations for the same
		x inputs, then all the warped noisy observations are used to fit the GP.
		Otherwise, only the mean of the outputs, for a given input x, is considered.
		Default is False.

	model_noise : string {'EGN',None}, optional
		Method to model the noise.
		If not None and if Y outputs contain multiple noisy observations for the same
		x inputs, then the nugget is estimated from the standard deviation of the multiple 
		outputs for a given input x.
		Default is None.

	isInt : boolean or (n_parameters) numpy array
		Specify which parameters are integers
		If isInt is a boolean, all parameters are assumed to have the same type.
		It is better to fix isInt=True rather than converting floating parameters as integers in the scoring
		function, because this would generate a discontinuous scoring function (whereas GP / GCP assume that
		the function is smooth)
	
	detailed_res : boolean, optional
		Specify if the method should return only the parameters and mean outputs or all the details, see below.


	
	Returns
	-------
	all_parameters : the parameters tested during the process.
		A list of length the number of models to use.
		For each model, this contains a ndarray of size (n_parameters_tested,n_features).

	all_raw_outputs : the detailed observations (if detailed_res == True ).
		A list of length the number of models to use.
		For each model, this contains a list of length the number of parameters tested during the process,
		for each parameter, the entry is a list containing all the (noisy) observations

	all_mean_outputs : the mean values of the outputs.
		A list of length the number of models to use.
		For each model, this contains a list of length the number of parameters tested during the process,
		and the values correspond to the mean of the (noisy) observations.

	all_std_outputs : the standard deviation of the observations (if detailed_res == True ).
		A list of length the number of models to use.
		For each model, this contains a list of length the number of parameters tested during the process,
		for each parameter, the entry is the standard deviation of the (noisy) observations.

	all_param_path : the path of the tested parameters.
		ndarray of size (n_models, n_total_iter,n_features), where n_total_iter is the sum of
		n_random_init, n_iter and n_final_iter.
		For each model, the ndarray stores the parameters tested and the order. The main difference
		between all_parameters is that all_parameters cannot contain twice the same parameter, but
		all_param_path can.
	"""
	

	#---------------------------- Init ----------------------------#
	GCP_upperBound_coef = 1.96

	n_parameters = parameter_bounds.shape[0]
	if(cluster_evol != 'constant'):
		GCP_args = [corr_kernel, 1,GCP_mapWithNoise,GCP_useAllNoisyY,model_noise,nugget,GCP_upperBound_coef]
	else:
		GCP_args = [corr_kernel, n_clusters,GCP_mapWithNoise,GCP_useAllNoisyY,model_noise,nugget,GCP_upperBound_coef]
			
	if(verbose):
		print 'parameter bounds :',parameter_bounds
		print 'n_parameters :', n_parameters
		print 'Nbr of final steps :', n_final_iter
		print 'GCP args :',GCP_args
		print 'Data size can vary between',data_size_bounds
		print 'Nugget :', nugget
		print 'GCP_upperBound_coef :',GCP_upperBound_coef

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
	param_path = []

	#-------------------- Random initialization --------------------#

	# sample n_random_init random parameters to initialize the process
	init_rand_candidates = utils.sample_random_candidates_for_init(n_random_init,parameter_bounds,data_size_bounds,isInt)
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
				utils.add_results(parameters,raw_outputs,mean_outputs,std_outputs,rand_candidate,new_output)
		param_path.append(rand_candidate)		

	all_parameters = []
	all_raw_outputs = []
	all_mean_outputs = []
	all_std_outputs = []
	all_param_path = []
	for i in range(nb_model):
		if(modelToRun[i]):
			all_parameters.append(np.copy(parameters))
			all_raw_outputs.append(list(raw_outputs))
			all_mean_outputs.append(list(mean_outputs))
			all_std_outputs.append(list(std_outputs))
			all_param_path.append(list(param_path))
		
		
	#------------------------ Smart Sampling ------------------------#
	
	i_mod_10 = 0

	for i in range(n_iter):
		if(i==20 and cluster_evol=='step'):
			GCP_args[1] = n_clusters

		if(i/10 > (i_mod_10+2) and cluster_evol=='variable'):
			GCP_args[0] = GCP_args[0]
			GCP_args[1] = min(GCP_args[1]+1,n_clusters_max)
			i_mod_10 += 3

		if(verbose):
			print('Step '+str(i))
			model_idx=0
			for k in range(nb_model):
				if(modelToRun[k]):
					print k,'current best output',np.max(all_mean_outputs[model_idx])
					model_idx += 1	
				
		rand_candidates = utils.sample_random_candidates(n_candidates,parameter_bounds,data_size_bounds,isInt)
		
		if(verbose):
			print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
		
		model_idx = 0
		for k in range(nb_model):
			if(modelToRun[k]):
				best_candidate = utils.find_best_candidate(k,
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
						utils.add_results(all_parameters[model_idx],all_raw_outputs[model_idx],\
							all_mean_outputs[model_idx],all_std_outputs[model_idx],\
								best_candidate,new_output)	
				all_param_path[model_idx].append(best_candidate)	

				model_idx += 1
					
				if(verbose):
					print k,'Test paramter:', best_candidate,' - ***** accuracy:',new_output
	

	#----------------- Last step : Try to find the max -----------------#

	if(verbose):
		print('\n*** Last step : try to find the best parameters ***')
		
	if(data_size_bounds is not None):
		data_size_bounds[0] = data_size_bounds[1]
		if(verbose):
			print('Fixed the data size at',data_size_bounds[0])
	
	for i in range(n_final_iter):

		if(verbose):
			print('Final step '+str(i))
		
		rand_candidates = utils.sample_random_candidates(n_candidates,parameter_bounds,data_size_bounds,isInt)
		
		if(verbose):
			print('Has sampled ' + str(rand_candidates.shape[0]) + ' random candidates')
		
		model_idx = 0
		for k in range(nb_model):
			if(modelToRun[k]):
				best_candidate = utils.find_best_candidate(k,
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
						utils.add_results(all_parameters[model_idx],all_raw_outputs[model_idx],\
							all_mean_outputs[model_idx],all_std_outputs[model_idx],\
								best_candidate,new_output)		
				
				all_param_path[model_idx].append(best_candidate)	

				model_idx += 1
					
				if(verbose):
					print k,'Test paramter:', best_candidate,' - ***** accuracy:',new_output
			

	#--------------------------- Final Result ---------------------------#

	best_parameters = []
	
	model_idx = 0
	for k in range(nb_model):
		if(modelToRun[k]):
			best_parameter_idx = np.argmax(all_mean_outputs[model_idx])
			print k,'Best parameters '+str(all_parameters[model_idx][best_parameter_idx]) + ' with output: ' + str(all_mean_outputs[model_idx][best_parameter_idx])
			if(data_size_bounds is not None):
				print k,'Best parameters for complete dataset'+ \
									str( (all_parameters[model_idx][all_parameters[model_idx][:,0] == data_size_bounds[1]])[best_parameter_idx2]) \
									+ ' with output: ' + \
									str( (all_mean_outputs[model_idx][all_parameters[model_idx][:,0] == data_size_bounds[1]])[best_parameter_idx2])

			model_idx += 1
	
	if(verbose):
		print '\n','n_parameters :', n_parameters
		print 'Nbr of final steps :', n_final_iter
		print 'GCP args :',GCP_args
		print 'Nugget :', nugget
		print 'GCP_upperBound_coef :',GCP_upperBound_coef

	if(detailed_res):
		return all_parameters , all_raw_outputs, all_mean_outputs, all_std_outputs, np.asarray(all_param_path)
	else:
		return all_parameters, all_mean_outputs

	

	
