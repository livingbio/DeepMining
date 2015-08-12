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
from random import randint, randrange
import search_utils as utils 

from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import check_cv, _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.base import is_classifier, clone

class SmartSearch(object):
	"""
    Examples
    --------
    >>> parameters = {'kernel' :  ['cat', ['rbf','poly']],
    ...   			   'd' : ['int', [1,3]],
    ...  			   'C' : ['float',[1,10])}

    """

	def __init__(self,
				parameters,
				estimator,
				model='GCP',
				score_format = 'cv',
				scoring=None,
				X=None,y=None,
				fit_params=None,
				refit=True, 
				cv=None,
				acquisition_function = 'UCB',
				corr_kernel= 'squared_exponential',
				n_clusters=1,
				n_clusters_max=5,
				cluster_evol = 'constant',
				GCP_mapWithNoise=False,
				GCP_useAllNoisyY=False,
				model_noise=None,
				n_iter=100,
				n_init=10,
				n_final_iter = 5,
				n_candidates = 500,
				nugget=1.e-10,
				verbose=True):

		self.parameters = parameters
		self.n_parameters = len(parameters)
		self.n_iter = n_iter
		self.n_init = n_init
		self.n_final_iter = n_final_iter
		self.n_candidates = n_candidates
		self.param_names = parameters.keys()
		self.param_isInt = np.array([ 0 if (parameters[k][0]=='float') else 1 for k in self.param_names ]) 
		self.param_bounds = np.zeros((self.n_parameters,2))
		self.verbose = verbose
		self.scoring = scoring
		self.estimator = estimator
		self.fit_params = fit_params if fit_params is not None else {}
		self.cv = cv
		self.X = X
		self.y = y

		self.model = model
		self.score_format = score_format # 'cv' or 'avg'
		self.acquisition_function = acquisition_function
		self.corr_kernel = corr_kernel
		self.n_clusters = n_clusters
		self.n_clusters_max = n_clusters_max
		self.cluster_evol = cluster_evol
		self.GCP_mapWithNoise = GCP_mapWithNoise
		self.GCP_useAllNoisyY = GCP_useAllNoisyY
		self.model_noise = model_noise		
		self.GCP_upperBound_coef = 1.96
		self.nugget = nugget

		if(cluster_evol != 'constant'):
			self.GCP_args = [corr_kernel, 1,GCP_mapWithNoise,GCP_useAllNoisyY,model_noise,nugget,self.GCP_upperBound_coef]
		else:
			self.GCP_args = [corr_kernel, n_clusters,GCP_mapWithNoise,GCP_useAllNoisyY,model_noise,nugget,self.GCP_upperBound_coef]
			
		if(callable(estimator)):
			self._callable_estimator = True
			if(verbose):
				print('Estimator is a callable and not an sklearn Estimator')
		else:
			self._callable_estimator = False

		if not self._callable_estimator:
			self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

		# init param_bounds
		for i in range(self.n_parameters):
			if(parameters[self.param_names[i]][0]=='cat'):
				self.param_bounds[i,0] = 0
				self.param_bounds[i,1] = len(parameters[self.param_names[i]][1])
			else:
				self.param_bounds[i] = np.array(parameters[self.param_names[i]][1])
				if(parameters[self.param_names[i]][0]=='int'):
					self.param_bounds[i,1] += 1

		# if(verbose):
		# print 'parameter bounds :',parameter_bounds
		# print 'n_parameters :', n_parameters
		# print 'Nbr of final steps :', n_final_iter
		# print 'GCP args :',GCP_args
		# print 'Data size can vary between',data_size_bounds
		# print 'Nugget :', nugget
		# print 'GCP_upperBound_coef :',GCP_upperBound_coef
		if(self.verbose):
			print(self.parameters)
			print(self.param_names)
			print(self.param_isInt)
			print(self.param_bounds)


	def vector_to_dict(self,vector_parameter):
		dict_parameter = dict.fromkeys(self.param_names)
		for i in range(self.n_parameters):
			if(self.parameters[self.param_names[i]][0]=='cat'):
				dict_parameter[self.param_names[i]] = (self.parameters[self.param_names[i]][1])[int(vector_parameter[i])]
			elif(self.parameters[self.param_names[i]][0]=='int'):
				dict_parameter[self.param_names[i]] = int(vector_parameter[i])
			else:
				dict_parameter[self.param_names[i]] = vector_parameter[i]

		return dict_parameter

	def score(self,test_parameter):
		if not self._callable_estimator:
	 		cv = check_cv(self.cv, self.X, self.y, classifier=is_classifier(self.estimator))
	 		cv_score = [ _fit_and_score(clone(self.estimator), self.X, self.y, self.scorer_,
							train, test, False, test_parameter,
							self.fit_params, return_parameters=True)
						for train, test in cv ]

			n_test_samples = 0
			mean_score = 0
			detailed_score = []
			for tmp_score, tmp_n_test_samples, _, _ in cv_score:
				detailed_score.append(tmp_score)
				tmp_score *= tmp_n_test_samples
				n_test_samples += tmp_n_test_samples
				mean_score += tmp_score
			mean_score /= float(n_test_samples)

			if(self.score_format == 'avg'):
				score = mean_score
			else: # format == 'cv'
				score = detailed_score


		else:
			if(self.score_format == 'avg'):
				score = [self.estimator(test_parameter)]
			else: # format == 'cv'
				score = self.estimator(test_parameter)

		return score


	def _fit(self):

		n_tested_parameters = 0
		tested_parameters = np.zeros((self.n_iter,self.n_parameters))
		cv_scores = []

		###    Initialize with random candidates    ### 
		init_candidates = utils.sample_candidates(self.n_init,self.param_bounds,self.param_isInt)
		self.n_init = init_candidates.shape[0]

		for i in range(self.n_init):
			dict_candidate = self.vector_to_dict(init_candidates[i,:])
			cv_score = self.score(dict_candidate)

			if(self.verbose):
				print ('Step ' + str(i) + ' - Hyperparameter ' + str(dict_candidate) + ' ' + str(np.mean(cv_score)))

			is_in,idx = utils.is_in_ndarray(init_candidates[i,:],tested_parameters[:n_tested_parameters,:])
			if not is_in:
				tested_parameters[n_tested_parameters,:] = init_candidates[i,:]
				cv_scores.append(cv_score)
				n_tested_parameters += 1
			else:
				if(self.verbose):
					print('Hyperparameter already tesed')
				cv_scores[idx] +=  cv_score


		###               Smart Search               ###    
		i_mod_10 = 0  
		for i in range(self.n_iter - self.n_init - self.n_final_iter):

			if(i==20 and cluster_evol=='step'):
				self.GCP_args[1] = n_clusters

			if(i/10 > (i_mod_10+2) and self.cluster_evol=='variable'):
				self.GCP_args[0] = self.GCP_args[0]
				self.GCP_args[1] = min(self.GCP_args[1]+1,self.n_clusters_max)
				i_mod_10 += 3
			
			# Sample candidates and predict their corresponding acquisition values
			candidates = utils.sample_candidates(self.n_candidates,self.param_bounds,self.param_isInt)

			# Model and retrieve the candidate that maximezes the acquisiton function
			best_candidate = utils.find_best_candidate(self.model,
														tested_parameters[:n_tested_parameters,:],
														cv_scores,
														self.GCP_args,
												 		candidates,
												 		self.verbose,
												 		self.acquisition_function)

			dict_candidate = self.vector_to_dict(best_candidate)
			cv_score = self.score(dict_candidate)

			if(self.verbose):
				print ('Step ' + str(i+self.n_init) + ' - Hyperparameter ' + str(dict_candidate) + ' ' + str(np.mean(cv_score)))

			is_in,idx = utils.is_in_ndarray(best_candidate,tested_parameters[:n_tested_parameters,:])
			if not is_in:
				tested_parameters[n_tested_parameters,:] = best_candidate
				cv_scores.append(cv_score)
				n_tested_parameters += 1
			else:
				if(self.verbose):
					print('Hyperparameter already tesed')
				cv_scores[idx] += cv_score


		###               Final steps               ###      
		self.acquisition_function = 'Simple'

		for i in range(self.n_final_iter):

			# Sample candidates and predict their corresponding acquisition values
			candidates = utils.sample_candidates(self.n_candidates,self.param_bounds,self.param_isInt)

			# Model and retrieve the candidate that maximezes the acquisiton function
			best_candidate = utils.find_best_candidate(self.model,
														tested_parameters[:n_tested_parameters,:],
														cv_scores,
														self.GCP_args,
												 		candidates,
												 		self.verbose,
												 		self.acquisition_function)

			dict_candidate = self.vector_to_dict(best_candidate)
			cv_score = self.score(dict_candidate)

			if(self.verbose):
				print ('Step ' + str(i+self.n_iter - self.n_final_iter) + ' - Hyperparameter ' + str(dict_candidate) + ' ' + str(np.mean(cv_score)))

			is_in,idx = utils.is_in_ndarray(best_candidate,tested_parameters[:n_tested_parameters,:])
			if not is_in:
				tested_parameters[n_tested_parameters,:] = best_candidate
				cv_scores.append(cv_score)
				n_tested_parameters += 1
			else:
				if(self.verbose):
					print('Hyperparameter already tesed')
				cv_scores[idx] += cv_score


		# compute the averages of CV results
		mean_scores = []
		for o in cv_scores:
			mean_scores.append(np.mean(o))

		# find the max
		best_idx = np.argmax(mean_scores)
		vector_best_param = tested_parameters[best_idx]
		best_parameter = self.vector_to_dict(vector_best_param)

		if(self.verbose):
			print ('\nTested ' + str(n_tested_parameters) + ' parameters')
			print ('Max cv score ' + str(mean_scores[best_idx]))
			print ('Best parameter ' + str(tested_parameters[best_idx]))
			print(best_parameter)

		return tested_parameters[:n_tested_parameters,:], cv_scores