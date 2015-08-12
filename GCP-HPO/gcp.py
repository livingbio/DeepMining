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

from __future__ import print_function


import numpy as np
from scipy import linalg, optimize
import math
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.gaussian_process import regression_models as regression
from scipy.stats import norm
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial import distance
from GCP_utils import *
import sklearn_utils as sk_utils
from scipy import integrate

MACHINE_EPSILON = np.finfo(np.double).eps



class GaussianCopulaProcess(BaseEstimator, RegressorMixin):
	"""The Gaussian Copula Process model class.

	Parameters
	----------
	regr : string or callable, optional
		A regression function returning an array of outputs of the linear
		regression functional basis. The number of observations n_samples
		should be greater than the size p of this basis.
		Default assumes a simple constant regression trend.
		Available built-in regression models are:

			'constant', 'linear', 'quadratic'

	corr : string or callable, optional
		A stationary autocorrelation function returning the autocorrelation
		between two points x and x'.
		Default assumes a squared-exponential autocorrelation model.
		Built-in correlation models are:

			'squared_exponential', 'exponential_periodic'

	beta0 : double array_like, optional
		The regression weight vector to perform Ordinary Kriging (OK).
		Default assumes Universal Kriging (UK) so that the vector beta of
		regression weights is estimated using the maximum likelihood
		principle.

	verbose : boolean, optional
		A boolean specifying the verbose level.
		Default is verbose = False.

	theta : double array_like, optional
		An array with shape (n_features, ) or (1, ).
		The parameters in the autocorrelation model.
		theta is the starting point for the maximum likelihood estimation of the
		best set of parameters.
		Default assumes isotropic autocorrelation model with theta0 = 1e-1.

	thetaL : double array_like, optional
		An array with shape matching theta0's.
		Lower bound on the autocorrelation parameters for maximum
		likelihood estimation.

	thetaU : double array_like, optional
		An array with shape matching theta0's.
		Upper bound on the autocorrelation parameters for maximum
		likelihood estimation.

	try_optimize : boolean, optional
		If True, perform maximum likelihood estimation to set the value of theta.
		Default is True.

	normalize : boolean, optional
		Input X and observations y are centered and reduced wrt
		means and standard deviations estimated from the n_samples
		observations provided.
		Default is normalize = True so that data is normalized to ease
		maximum likelihood estimation.

	reNormalizeY : boolean, optional
		Normalize the warped Y values, ie. the values mapping(Yt), before
		fitting a GP.
		Default is False.

	n_clusters : int, optional
		If n_clusters > 1, a latent model is built by clustering the data with
		K-Means, into n_clusters clusters.
		Default is 1.

	coef_latent_mapping : float, optional
		If n_clusters > 1, this coefficient is used to interpolate the mapping
		function on the whole space from the mapping functions learned on each
		cluster. This acts as a smoothing parameter : if coef_latent_mapping == 0.,
		each cluster contributes equally, and the greater it is the fewer
		mapping(x,y) takes into account the clusters in which x is not. 
		Default is 0.5.

	mapWithNoise : boolean, optional
		If True and if Y outputs contain multiple noisy observations for the same
		x inputs, then all the noisy observations are used to compute Y's distribution
		and learn the mapping function.
		Otherwise, only the mean of the outputs, for a given input x, is considered.
		Default is False.

	useAllNoisyY : boolean, optional
		If True and if Y outputs contain multiple noisy observations for the same
		x inputs, then all the warped noisy observations are used to fit the GP.
		Otherwise, only the mean of the outputs, for a given input x, is considered.
		Default is False.

	model_noise : string, optional
		Method to model the noise.
		If not None and if Y outputs contain multiple noisy observations for the same
		x inputs, then the nugget (see below) is estimated from the standard
		deviation of the multiple outputs for a given input x. Precisely the nugget
		is multiplied by 100 * std (as data is usually normed and noise is usually
		of the order of 1%).
		Default is None, methods currently available are :

			'EGN' (Estimated Gaussian Noise)

	nugget : double or ndarray, optional
		Introduce a nugget effect to allow smooth predictions from noisy
		data.  If nugget is an ndarray, it must be the same length as the
		number of data points used for the fit.
		The nugget is added to the diagonal of the assumed training covariance;
		in this way it acts as a Tikhonov regularization in the problem.  In
		the special case of the squared exponential correlation function, the
		nugget mathematically represents the variance of the input values.
		Default assumes a nugget close to machine precision for the sake of
		robustness (nugget = 10. * MACHINE_EPSILON).

	random_start : int, optional
		The number of times the Maximum Likelihood Estimation should be
		performed from a random starting point.
		The first MLE always uses the specified starting point (theta0),
		the next starting points are picked at random according to an
		exponential distribution (log-uniform on [thetaL, thetaU]).
		Default is 5.

	random_state: integer or numpy.RandomState, optional
		The generator used to shuffle the sequence of coordinates of theta in
		the Welch optimizer. If an integer is given, it fixes the seed.
		Defaults to the global numpy random number generator.


	Attributes
	----------
	`theta_`: array
		Specified theta OR the best set of autocorrelation parameters (the \
		sought maximizer of the reduced likelihood function).

	`reduced_likelihood_function_value_`: array
		The optimal reduced likelihood function value.
	
	`centroids` : array of shape (n_clusters,n_features)
		If n_clusters > 1, the array of the clusters' centroid.

	`density_functions` : list of callable
		List of length n_clusters, containing the density estimations \
		of the outputs on each cluster.

	`mapping` : callable
		The mapping function such that mapping(Yt) has a gaussian \
		distribution, where Yt is the output. \
		The mapping is learned based on the KDE estimation of Yt's distribution
		
	`mapping_inv` : callable
		The inverse mapping numerically computed by binomial search,\
		such that mapping_inv[mapping(.)] == mapping[mapping_inv(.)] == id
	
	`mapping_derivative` : callable
		The derivative of the mapping function.		


	Notes
	-----
	This code is based on scikit-learn's GP implementation.

	References
	----------
	On Gaussian processes :
	.. [NLNS2002] `H.B. Nielsen, S.N. Lophaven, H. B. Nielsen and J.
		Sondergaard.  DACE - A MATLAB Kriging Toolbox.` (2002)
		http://www2.imm.dtu.dk/~hbn/dace/dace.pdf

	.. [WBSWM1992] `W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell,
		and M.D.  Morris (1992). Screening, predicting, and computer
		experiments.  Technometrics, 34(1) 15--25.`
		http://www.jstor.org/pss/1269548

	On Gaussian Copula processes :
	.. Wilson, A. and Ghahramani, Z. Copula processes. In Advances in NIPS 23,
		pp. 2460-2468, 2010
	"""

	_regression_types = {
		'constant': regression.constant,
		'linear': regression.linear,
		'quadratic': regression.quadratic}


	def __init__(self, regr='constant', 
				 corr='exponential_periodic',
				 verbose=False,
				 theta=np.asarray([40,30,20,.05,1.,1.,.5,.001,10.]),
                 thetaL=np.asarray([10.,1.,1.,.0001,1.,0.1,0.1,0.0001,1.]),
                 thetaU=np.asarray([90,50,0.1,1.,1000.,10.,1.,.01,100.]), 
				 try_optimize=True,
				 random_start=5, 
				 normalize=True,
				 reNormalizeY = False,
				 n_clusters = 1,
				 coef_latent_mapping = 0.5,
				 mapWithNoise=False,
				 useAllNoisyY=False,
				 model_noise=None,
				 nugget=10. * MACHINE_EPSILON,
				 random_state=None):
 
		self.regr = regr
		self.beta0 = None
		self.verbose = verbose
		self.theta = theta
		self.thetaL = thetaL
		self.thetaU = thetaU
		self.normalize = normalize
		self.reNormalizeY = reNormalizeY
		self.nugget = nugget
		self.optimizer = 'fmin_cobyla'
		self.random_start = random_start
		self.random_state = random_state
		self.try_optimize = try_optimize
		self.n_clusters = n_clusters
		self.density_functions = None
		self.coef_latent_mapping = coef_latent_mapping
		self.mapWithNoise = mapWithNoise
		self.useAllNoisyY =useAllNoisyY
		if (corr == 'squared_exponential'):
			self.corr = sq_exponential
			self.theta = np.asarray([0.1])
			self.thetaL = np.asarray([0.001])
			self.thetaU = np.asarray([10.])
		else:
			self.corr = exponential_periodic
		self.model_noise = model_noise
		if(self.model_noise == 'EGN' and self.useAllNoisyY):
			self.useAllNoisyY = False

	def mapping(self,x,t,normalize=False):
		if(normalize):
			t = (t-self.raw_y_mean) / self.raw_y_std
		v = 0.
		# if t is too big, F_est(t) will be too close to 1
		# this is an issue as in that case norm.ppf( F_est(t) ) will return Nan
		# so we force F_est(t) to be smaller than 0.999999998 which corresponds to
		# norm.ppf( F_est(t) ) being smaller than 6.3
		# Normally doing this shouldn't cause any problem. But such extrem t values can
		# be queried by the binomial search to invert the mapping (mapping_inv)
		if( t < 2047483647):
			if(self.n_clusters > 1):
				coefs = np.ones(self.n_clusters)
				val = np.zeros(self.n_clusters)
				for w in range(self.n_clusters):
					## coefficients are :
					#	 exp{  - sum [ (d_i /std_i) **2 ]  }
					coefs[w] =  np.exp(- np.sum( (self.coef_latent_mapping*(x -self.centroids[w])/self.clusters_std)**2. ) )
					temp  =  self.density_functions[w].integrate_box_1d(self.low_bound, t)
					# if temp is too close to 1, norm.ppf(temp) == Nan
					# if temp == 0, norm.ppf(temp) == -inf
					temp = min(0.9999999999999999,temp)
					if(temp == 0):
						temp = 1e-10
					val[w] = ( norm.ppf(temp) ) 
				
				s = np.sum(coefs)
				if(s != 0):
					val = val * coefs
				else:
					s = self.n_clusters
				val = val / s
				v = np.sum(val)
				
			else:
				temp = self.density_functions[0].integrate_box_1d(self.low_bound, t)
				# if temp is too close to 1, norm.ppf(temp) == Nan
				# if temp == 0, norm.ppf(temp) == -inf
				temp = min(0.9999999999999999,temp)
				if(temp == 0):
					temp = 1e-10
				v = norm.ppf(temp)
		else:
			v = 8.3
			
		return [v]
		

	def mapping_inv(self,x,t):
		if(t<= 8.2):
			def map(t):
				return self.mapping(x,t)
			lo, hi = find_bounds(map, t)
			res = binary_search(map, t, lo, hi)				
			return [res]
		else:
			return [2047483647]


	def mapping_derivative(self,x,t,normalize=False):
		# our mapping function is constant for t values greater than 2047483646
		# so for such values mapping_derivate should be null
		if(normalize):
			t = (t-self.raw_y_mean) / self.raw_y_std
		v = 0.
		if( t < 2047483646):
			if(self.n_clusters > 1):
				coefs = np.ones(self.n_clusters)
				val = np.zeros(self.n_clusters)
				for w in range(self.n_clusters):
					## coefficients are :
					#	 exp{  - sum [ (d_i /std_i) **2 ]  }
					coefs[w] =  np.exp(- np.sum( (self.coef_latent_mapping*(x -self.centroids[w])/self.clusters_std)**2. ) )
					# computing Psi_i (t) 
					temp =  self.density_functions[w].integrate_box_1d(self.low_bound, t)
					temp = min(0.9999999999999999,temp)
					temp = max( temp, 1e-10)
					temp = norm.ppf(temp)
					# pdf( Psi_i (t))
					temp = norm.pdf(temp)
					# d_est_i / pdf(...)
					temp = self.density_functions[w](t) / temp
					val[w] = temp
				s = np.sum(coefs)
				if(s != 0):
					val = val * coefs
				else:
					s = self.n_clusters
				val = val / s
				v = np.sum(val)

			else:
				temp =  self.density_functions[0].integrate_box_1d(self.low_bound, t)
				temp = min(0.9999999999999999,temp)
				temp = max( temp, 1e-10)
				temp = norm.ppf(temp)
				# pdf( Psi (t))
				temp = norm.pdf(temp)
				# d_est / pdf(...)
				v = self.density_functions[0](t) / temp
		else:
			v = 0.
			
		return (v/self.raw_y_std)	


	def integrate_prediction(self,mu,sigma,x,lb,ub):
		# utility function for the predict method
		def f_to_integrate(u):
			temp = norm.pdf(self.mapping(x,u,normalize=True),loc=mu,scale=sigma)
			temp = temp * (u * self.mapping_derivative(x,u,normalize=True) )
			return temp
		return(integrate.quad(f_to_integrate,lb,ub,epsrel =0.000000001)[0])


	def predicted_RV(self,mu,sigma,x):
		# utility function for the predict method
		def f_to_integrate(u):
			temp = norm.pdf(self.mapping(x,u,normalize=True),loc=mu,scale=sigma)
			temp = temp * (u * self.mapping_derivative(x,u,normalize=True) )
			return temp
		return f_to_integrate


	def init_mappings(self):
		# We assume y is one-dimensional
	
		# Perform KMeans and store results
		if(self.n_clusters > 1):

			clustering_pending = True
			while(clustering_pending and self.n_clusters >1 ):
				clustering_pending = False
				kmeans = KMeans(n_clusters=self.n_clusters)
				all_data = []
				for i in range(self.X.shape[0]):
					all_data.append( np.concatenate((self.X[i],self.raw_y[i])))
				all_data = np.asarray(all_data)
				windows_idx = kmeans.fit_predict(all_data)
				self.centroids = kmeans.cluster_centers_[:,:-1]
		
				if(self.verbose):
					print('All data shape :',all_data.shape)
					print ("Centroids")
					print (self.X_std*self.centroids + self.X_mean,self.raw_y_std*kmeans.cluster_centers_[:,-1] +self.raw_y_mean)
				
				# Compute the density function for each sub-window
				density_functions = []
				clusters_std = []
				if(self.detailed_raw_y is not None and self.mapWithNoise):
					detailed_windows_idx =reshape_cluster_labels(windows_idx,self.detailed_X)
				for w in range(self.n_clusters):
					if(self.detailed_raw_y is not None and self.mapWithNoise):
						cluster_points_y_values = np.copy((self.detailed_raw_y[ detailed_windows_idx == w])[:,0])
					else:
						cluster_points_y_values = np.copy((self.raw_y[ windows_idx == w])[:,0])
					clusters_std.append( np.std( self.X[ windows_idx == w], axis=0) ) ### this is a (Xdim) array
					if(self.verbose):
						print('cluster '+str(w)+' size ' + str(cluster_points_y_values.shape))
					if(cluster_points_y_values.shape[0] == 1):
						clustering_pending = True
					else:
						density_functions.append(stats.gaussian_kde(cluster_points_y_values) )
				
				if(clustering_pending):
					print ('Fail to build ' + str(self.n_clusters) + ' clusters')
					self.n_clusters -= 1

				else:
					density_functions = np.asarray( density_functions)
					self.density_functions = density_functions
					clusters_std = np.asarray(clusters_std)
					clusters_std[clusters_std==0] = 1.
					self.clusters_std = clusters_std

					if(self.verbose):
						print('---STD---')
						print(clusters_std)	

			if(clustering_pending):
				# n_cluster == 1
				if(self.detailed_raw_y is not None and self.mapWithNoise):
					self.density_functions = np.asarray( [ stats.gaussian_kde(self.detailed_raw_y[:,0]) ])
				else:
					self.density_functions = np.asarray( [ stats.gaussian_kde(self.raw_y[:,0]) ])	

		else:
			if(self.detailed_raw_y is not None and self.mapWithNoise):
				self.density_functions = np.asarray( [ stats.gaussian_kde(self.detailed_raw_y[:,0]) ])
			else:
				self.density_functions = np.asarray( [ stats.gaussian_kde(self.raw_y[:,0]) ])
		
		
	def update_copula_params(self):

		size = self.raw_y.shape[0]
		y = [ self.mapping(self.X[i],self.raw_y[i]) for i in range(size)]
		y = np.asarray(y)
		
		# Normalize data
		if(self.reNormalizeY):
			y_mean = np.mean(y, axis=0)
			y_std = np.std(y, axis=0)
			y_std[y_std == 0.] = 1.
		else:
			y_mean = 0.
			y_std = 1.
		y = (y - y_mean) / y_std

		if(self.obs_noise is not None and self.model_noise == 'EGN' ):
			self.nugget = self.nugget *( ( 10. * self.obs_noise ) ** 2. )

		# Calculate matrix of distances D between samples
		D, ij = l1_cross_distances(self.X)
		#if (np.min(np.sum(D, axis=1)) == 0.):
		#	raise Exception("Multiple input features cannot have the same"
		#					" target value.")

		n_samples = self.X.shape[0]
		# Regression matrix and parameters
		F = self.regr(self.X)
		n_samples_F = F.shape[0]
		if F.ndim > 1:
			p = F.shape[1]
		else:
			p = 1
		if n_samples_F != n_samples:
			raise Exception("Number of rows in F and X do not match. Most "
							"likely something is going wrong with the "
							"regression model.")
		if p > n_samples_F:
			raise Exception(("Ordinary least squares problem is undetermined "
							 "n_samples=%d must be greater than the "
							 "regression model size p=%d.") % (n_samples, p))

		self.D = D
		self.ij = ij
		self.F = F
		self.y = y
		self.y_mean, self.y_std = y_mean, y_std
			
			
	def fit(self, X, y,detailed_y_obs=None,obs_noise=None):
		"""
		The Gaussian Copula Process model fitting method.

		Parameters
		----------
		X : double array_like
			An array with shape (n_samples, n_features) with the input at which
			observations were made.

		y : double array_like
			An array with shape (n_samples, ) or shape (n_samples, n_targets)
			with the observations of the output to be predicted.
			Currently only 1D targets are supported.

		detailed_y_obs : double list of list
			A list of length n_samples where entry at position i corresponds to 
			the mutiple noisy observations (given as a list) of the input value X[i,:], 
			and whose mean is y[i].
			If not None, it can be used to learn the mapping function or fit the GP,
			see parameters mapWithNoise and useAllNoisyY of the GaussianCopulaProcess class.

		obs_noise : double array
			An array of shape (n_samples,) corresponding to the estimated noise
			in the observed y outputs.
			If not None, it can be used to model the noise with the Estimated
			Gaussian Noise method, see the model_noise parameter of the 
			GaussianCopulaProcess class.

		Returns
		-------
		gcp : self
			A fitted Gaussian Copula Process model object awaiting data to perform
			predictions.
		"""
		# Run input checks
		self._check_params()
		X = sk_utils.array2d(X)
		# Check if all CV obs are given
		# and if so, convert this list of list to array
		if(detailed_y_obs is not None):
			detailed_X,detailed_raw_y = listOfList_toArray(X,detailed_y_obs)	

		y = np.asarray(y)
		self.y_ndim_ = y.ndim
		if y.ndim == 1:
			y = y[:, np.newaxis]
			if(detailed_y_obs is not None):
				detailed_raw_y = detailed_raw_y[:, np.newaxis]
		else:
			print('Warning: code is not ready for y outputs with dimension > 1')
		
		# Reshape theta if it is one dimensional and X is not
		x_dim = X.shape[1]
		#if not(self.theta.ndim == 1):
		#	print('Warning : theta has not the right shape')
		self.theta = (np.ones((x_dim,self.theta.shape[0])) * self.theta ).T
		self.thetaL = (np.ones((x_dim,self.thetaL.shape[0])) * self.thetaL ).T
		self.thetaU = (np.ones((x_dim,self.thetaU.shape[0])) * self.thetaU ).T
		#print('theta has new shape '+str(self.theta.shape))
			
		self.random_state = check_random_state(self.random_state)		
		X, y = sk_utils.check_arrays(X, y)

		# Check shapes of DOE & observations
		n_samples, n_features = X.shape
		_, n_targets = y.shape

		# Run input checks
		self._check_params(n_samples)

		# Normalize data or don't
		if self.normalize:
			X_mean = np.mean(X, axis=0)
			X_std = np.std(X, axis=0)
			X_std[X_std == 0.] = 1.
			X = (X - X_mean) / X_std

			raw_y_mean = np.mean(y, axis=0)
			raw_y_std = np.std(y, axis=0)
			raw_y_std[raw_y_std == 0.] = 1.
			y = (y - raw_y_mean) / raw_y_std
			
			if(obs_noise is not None):
				obs_noise = obs_noise / raw_y_std

			if(detailed_y_obs is not None):
				detailed_raw_y = (detailed_raw_y - raw_y_mean) / raw_y_std
				detailed_X = (detailed_X - X_mean) / X_std
		else:
			X_mean = np.zeros(1)
			X_std = np.ones(1)
		
		self.raw_y_min = np.min(y)		
		self.raw_y_max = np.max(y)

		# Set attributes

		if(detailed_y_obs is not None):
			self.detailed_raw_y = detailed_raw_y
			self.detailed_X = detailed_X
		else:
			self.detailed_raw_y = None			
			self.detailed_X = None			
		
		if(self.detailed_raw_y is not None and self.mapWithNoise and self.useAllNoisyY ):
			self.X = self.detailed_X
			self.raw_y = self.detailed_raw_y
			
			self.raw_y_mean = raw_y_mean
			self.raw_y_std = raw_y_std
			self.low_bound = np.min([-500., 5. * np.min(y)])
			self.X_mean, self.X_std = X_mean, X_std

			# initialize mapping only if needed, i.e. it hasn't be done 
			# yet of if we want to optimize the GCP hyperparameters
			if (self.try_optimize or (self.density_functions is None)):
				self.init_mappings()
		
		elif(self.detailed_raw_y is not None and self.useAllNoisyY ):
			self.X = X
			self.raw_y = y

			self.raw_y_mean = raw_y_mean
			self.raw_y_std = raw_y_std
			self.low_bound = np.min([-500., 5. * np.min(y)])
			self.X_mean, self.X_std = X_mean, X_std

			# initialize mapping only if needed, i.e. it hasn't be done 
			# yet of if we want to optimize the GCP hyperparameters
			if (self.try_optimize or (self.density_functions is None)):
				self.init_mappings()

			self.X = self.detailed_X
			self.raw_y = self.detailed_raw_y

		else:
			self.X = X
			self.raw_y = y

			self.raw_y_mean = raw_y_mean
			self.raw_y_std = raw_y_std
			self.low_bound = np.min([-500., 5. * np.min(y)])
			self.X_mean, self.X_std = X_mean, X_std

			# initialize mapping only if needed, i.e. it hasn't be done 
			# yet of if we want to optimize the GCP hyperparameters
			if (self.try_optimize or (self.density_functions is None)):
				self.init_mappings()

		self.obs_noise = obs_noise
		self.update_copula_params()
		
		if self.try_optimize:
		    # Maximum Likelihood Estimation of the parameters
			if self.verbose:
				print("Performing Maximum Likelihood Estimation of the "
					  "autocorrelation parameters...")
			self.theta, self.reduced_likelihood_function_value_, par = \
				self._arg_max_reduced_likelihood_function()
			if np.isinf(self.reduced_likelihood_function_value_):
				raise Exception("Bad parameter region. "
								"Try increasing upper bound")
		else:
			# Given parameters
			if self.verbose:
				print("Given autocorrelation parameters. "
					  "Computing Gaussian Process model parameters...")
			self.reduced_likelihood_function_value_, par = \
				self.reduced_likelihood_function()
			if np.isinf(self.reduced_likelihood_function_value_):
				raise Exception("Bad point. Try increasing theta0.")

		self.beta = par['beta']
		self.gamma = par['gamma']
		self.sigma2 = par['sigma2']
		self.C = par['C']
		self.Ft = par['Ft']
		self.G = par['G']

		return self

	def predict(self, X, eval_MSE=False, transformY=True, returnRV=False, integratedPrediction= False, eval_confidence_bounds=False,coef_bound=1.96, batch_size=None):
		"""
		This function evaluates the Gaussian Process model at x.

		Parameters
		----------
		X : array_like
			An array with shape (n_eval, n_features) giving the point(s) at
			which the prediction(s) should be made.

		eval_MSE : boolean, optional
			A boolean specifying whether the Mean Squared Error should be
			evaluated or not.
			Default assumes evalMSE = False and evaluates only the BLUP (mean
			prediction).

		transformY : boolean, optional
			A boolean specifying if the predicted values should correspond to
			the same space as the data given to the fit method, or to the
			warped space (in which the GP is fitted).
			Default is True. Setting to False can be useful to compute the Expected
			Improvement in an optimization process.

		returnRV : boolean, optional
			A boolean specifying if the method should return the predicted random variables
			at x instead of a float number.
			Default is False.

		integratedPrediction : boolean, optional
			A boolean specifying if the method should return the fully Bayesian
			prediction, ie compute the expectation given the posterior in the
			original space. If False, the returned value is the inverse value
			(by the mapping function) of the GP prediction. This is much more faster
			as the integratedPrediction needs to numerically compute the integral.
			Default is False.

		eval_confidence_bounds : boolean, optional
			A boolean specifying if the method should return the confidence bounds.
			Because of the non-linearity of the mapping function, this cannot be computed
			directly with the MSE, but needs to invert the mapping function.
			Default is False. If True, coef_bound specifies the boundary to compute.

		coef_bound : float, optional
			A float specifying the confidence bounds to compute. Upper and lower
			confidence bounds are computed as the inverse of m + coef_bound*sigma
			where m and sigma are the mean and the std of the posterior in the GP
			space.
			Default is 1.96 which corresponds to the 95% confidence bounds.

		batch_size : integer, optional
			An integer giving the maximum number of points that can be
			evaluated simultaneously (depending on the available memory).
			Default is None so that all given points are evaluated at the same
			time.

		Returns
		-------
		y : array_like, shape (n_samples,)
			Prediction at x.

		MSE : array_like, optional (if eval_MSE == True)
			Mean Squared Error at x.

		LCB : array_like, optional (if eval_confidence_bounds == True)
			Lower confidence bound.

		UCB : array_like, optional (if eval_confidence_bounds == True)
			Upper confidence bound.
		"""

		# Check input shapes
		X = sk_utils.array2d(X)
		n_eval, _ = X.shape
		n_samples, n_features = self.X.shape
		n_samples_y, n_targets = self.y.shape

		if(n_targets > 1):
			raise ValueError('More than one target in the Y outputs. \
							  Currently only 1D outputs are handled')

		# Run input checks
		self._check_params(n_samples)

		if X.shape[1] != n_features:
			raise ValueError(("The number of features in X (X.shape[1] = %d) "
							  "should match the number of features used "
							  "for fit() "
							  "which is %d.") % (X.shape[1], n_features))

		# Normalize input
		if self.normalize:
			X = (X - self.X_mean) / self.X_std
			
		# Initialize output
		y = np.zeros(n_eval)
		if eval_MSE:
			MSE = np.zeros(n_eval)

		# Get pairwise componentwise L1-distances to the input training set
		dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
		# Get regression function and correlation
		f = self.regr(X)
		r = self.corr(self.theta, dx).reshape(n_eval, n_samples)

		# Scaled predictor
		y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

		# Predictor
		y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

		# transform the warped y, modeled as a Gaussian, to the real y
		size = y.shape[0]
		warped_y = np.copy(y)
		
		if(transformY):
			if( np.sum([ y[i][0] > 8.2 for i in range(size)]) >0):
				print('Warning : mapping_inversion failed')
			real_y = [ self.mapping_inv(X[i],y[i][0]) for i in range(size)]
			real_y = self.raw_y_std * np.asarray(real_y) +self.raw_y_mean
			y = real_y.reshape(n_eval, n_targets)
		
		if self.y_ndim_ == 1:
			y = y.ravel()
			warped_y = warped_y.ravel()

		# Mean Squared Error
		if eval_MSE:
			C = self.C
			if C is None:
				# Light storage mode (need to recompute C, F, Ft and G)
				if self.verbose:
					print("This GaussianProcess used 'light' storage mode "
						  "at instantiation. Need to recompute "
						  "autocorrelation matrix...")
				reduced_likelihood_function_value, par = \
					self.reduced_likelihood_function()
				self.C = par['C']
				self.Ft = par['Ft']
				self.G = par['G']

			rt = linalg.solve_triangular(self.C, r.T, lower=True)

			if self.beta0 is None:
				# Universal Kriging
				u = linalg.solve_triangular(self.G.T,
											np.dot(self.Ft.T, rt) - f.T)
			else:
				# Ordinary Kriging
				u = np.zeros((n_targets, n_eval))

			MSE = np.dot(self.sigma2.reshape(n_targets, 1),
						 (1. - (rt ** 2.).sum(axis=0)
						  + (u ** 2.).sum(axis=0))[np.newaxis, :])
			MSE = np.sqrt((MSE ** 2.).sum(axis=0) / n_targets)

			# Mean Squared Error might be slightly negative depending on
			# machine precision: force to zero!
			MSE[MSE < 0.] = 0.

			if self.y_ndim_ == 1:
				MSE = MSE.ravel()
				sigma = np.sqrt(MSE)
				if(returnRV):
					return [ self.predicted_RV([warped_y[i]],sigma[i],X[i]) for i in range(size)]
				else:
					if(eval_confidence_bounds):
						if not(transformY):
							print('Warning, transformY set to False but trying to evaluate conf bounds')
						warped_y_with_boundL = warped_y - coef_bound * sigma
						warped_y_with_boundU = warped_y + coef_bound * sigma
						pred_with_boundL = self.raw_y_std * np.asarray( [ self.mapping_inv(X[i],warped_y_with_boundL[i])[0] for i in range(size) ] ) +self.raw_y_mean
						pred_with_boundU =  self.raw_y_std * np.asarray( [ self.mapping_inv(X[i],warped_y_with_boundU[i])[0] for i in range(size)] ) +self.raw_y_mean
						
						if(integratedPrediction):
							lb = self.raw_y_min - 3.*(self.raw_y_max-self.raw_y_min)
							ub = self.raw_y_max + 3.*(self.raw_y_max-self.raw_y_min)
							print(lb,ub)
							integrated_real_y = [ self.integrate_prediction([warped_y[i]],sigma[i],X[i],lb,ub) for i in range(size)]
							integrated_real_y =  np.asarray(integrated_real_y)
							print('Integrated prediction')
							return integrated_real_y,MSE,pred_with_boundL,pred_with_boundU

						else:
							return y,MSE,pred_with_boundL,pred_with_boundU

						
					else:
						return y, MSE
			
			else:
				return y, MSE

		else:
			return y


	def reduced_likelihood_function(self, theta=None):
		"""
		This function determines the BLUP parameters and evaluates the reduced
		likelihood function for the given autocorrelation parameters theta.

		Maximizing this function wrt the autocorrelation parameters theta is
		equivalent to maximizing the likelihood of the assumed joint Gaussian
		distribution of the observations y evaluated onto the design of
		experiments X.

		Parameters
		----------
		theta : array_like, optional
			An array containing the autocorrelation parameters at which the
			Gaussian Process model parameters should be determined.
			Default uses the built-in autocorrelation parameters
			(ie ``theta = self.theta_``).

		Returns
		-------
		reduced_likelihood_function_value : double
			The value of the reduced likelihood function associated to the
			given autocorrelation parameters theta.

		par : dict
			A dictionary containing the requested Gaussian Process model
			parameters:

				sigma2
						Gaussian Process variance.
				beta
						Generalized least-squares regression weights for
						Universal Kriging or given beta0 for Ordinary
						Kriging.
				gamma
						Gaussian Process weights.
				C
						Cholesky decomposition of the correlation matrix [R].
				Ft
						Solution of the linear equation system : [R] x Ft = F
				G
						QR decomposition of the matrix Ft.
		"""
		
		if theta is None:
			# Use built-in autocorrelation parameters
			theta = self.theta

		# Initialize output
		reduced_likelihood_function_value = - np.inf
		par = {}

		# Retrieve data
		n_samples = self.X.shape[0]
		D = self.D
		ij = self.ij
		F = self.F

		if D is None:
			# Light storage mode (need to recompute D, ij and F)
			D, ij = l1_cross_distances(self.X)
			#if (np.min(np.sum(D, axis=1)) == 0.):
			#	raise Exception("Multiple X are not allowed")
			F = self.regr(self.X)

		# Set up R
		r = self.corr(theta, D)
		R = np.eye(n_samples) * (1. + self.nugget)
		R[ij[:, 0], ij[:, 1]] = r
		R[ij[:, 1], ij[:, 0]] = r

		# Cholesky decomposition of R
		try:
			C = linalg.cholesky(R, lower=True)
		except linalg.LinAlgError:
			return reduced_likelihood_function_value, par

		# Get generalized least squares solution
		Ft = linalg.solve_triangular(C, F, lower=True)
		try:
			Q, G = linalg.qr(Ft, econ=True)
		except:
			#/usr/lib/python2.6/dist-packages/scipy/linalg/decomp.py:1177:
			# DeprecationWarning: qr econ argument will be removed after scipy
			# 0.7. The economy transform will then be available through the
			# mode='economic' argument.
			Q, G = linalg.qr(Ft, mode='economic')
			pass

		sv = linalg.svd(G, compute_uv=False)
		rcondG = sv[-1] / sv[0]
		if rcondG < 1e-10:
			# Check F
			sv = linalg.svd(F, compute_uv=False)
			condF = sv[0] / sv[-1]
			if condF > 1e15:
				raise Exception("F is too ill conditioned. Poor combination "
								"of regression model and observations.")
			else:
				# Ft is too ill conditioned, get out (try different theta)
				return reduced_likelihood_function_value, par

		Yt = linalg.solve_triangular(C, self.y, lower=True)
		if self.beta0 is None:
			# Universal Kriging
			beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
		else:
			# Ordinary Kriging
			beta = np.array(self.beta0)

		rho = Yt - np.dot(Ft, beta)
		sigma2 = (rho ** 2.).sum(axis=0) / n_samples
		# The determinant of R is equal to the squared product of the diagonal
		# elements of its Cholesky decomposition C
		detR = (np.diag(C) ** (2. / n_samples)).prod()

		# Compute/Organize output
		reduced_likelihood_function_value = - sigma2.sum() * detR
		par['sigma2'] = sigma2 * self.y_std ** 2.
		par['beta'] = beta
		par['gamma'] = linalg.solve_triangular(C.T, rho)
		par['C'] = C
		par['Ft'] = Ft
		par['G'] = G

		return reduced_likelihood_function_value, par

	def _arg_max_reduced_likelihood_function(self):
		"""
		This function estimates the autocorrelation parameters theta as the
		maximizer of the reduced likelihood function.
		(Minimization of the opposite reduced likelihood function is used for
		convenience)

		Parameters
		----------
		self : All parameters are stored in the Gaussian Process model object.

		Returns
		-------
		optimal_theta : array_like
			The best set of autocorrelation parameters (the sought maximizer of
			the reduced likelihood function).

		optimal_reduced_likelihood_function_value : double
			The optimal reduced likelihood function value.

		optimal_par : dict
			The BLUP parameters associated to thetaOpt.
		"""
		
		# Initialize output
		best_optimal_theta = []
		best_optimal_rlf_value = []
		best_optimal_par = []

		if self.verbose:
			print("The chosen optimizer is: " + str(self.optimizer))
			if self.random_start > 1:
				print(str(self.random_start) + " random starts are required.")

		percent_completed = 0.

		if self.optimizer == 'fmin_cobyla':

			def minus_reduced_likelihood_function(x):
				x_reshaped = theta_backToRealShape(x,self.theta.shape)
				return - self.reduced_likelihood_function(
					theta=10. ** x_reshaped)[0]
						
			constraints = []
			# http://stackoverflow.com/questions/25985868/scipy-why-isnt-cobyla-respecting-constraint
			
			# Cobyla takes only one dimensional array
			conL = np.log10(theta_toOneDim(self.thetaL))
			conU = np.log10(theta_toOneDim(self.thetaU))
			
			def kernel_coef(x):
				return(100. - ((10. ** x[0]) + (10. ** x[1]) + (10. ** x[2]) ))
			
			if(self.theta.shape[0] > 1):
				n_idx = self.theta.size -3*(self.theta.shape[1]-1)	
			else:
				n_idx = self.theta.size
			for idx in range(n_idx):
				lower = conL[idx]
				upper = conU[idx]
				constraints.append(lambda x, a=lower, i=idx: x[i] - a)
				constraints.append(lambda x, b=upper, i=idx: b - x[i])
			
			if(self.theta.shape[0] > 1):
				constraints.append(kernel_coef)
			
			k=0
			k2 = 0
			while( (k < self.random_start) and (k2 < 50)):
					
				if (k == 0 and k2 ==0):
					# Use specified starting point as first guess
					theta0 = theta_toOneDim(self.theta)

				else:
					# Generate a random starting point log10-uniformly
					# distributed between bounds
					log10theta0 = np.log10(self.thetaL) \
						+ self.random_state.rand(self.theta.size).reshape(
							self.theta.shape) * np.log10(self.thetaU
														  / self.thetaL)
					theta0 = 10. ** theta_toOneDim(log10theta0)
					
				# Run Cobyla
				try:
					params= np.log10(theta0)
					if self.verbose:
						print('try '+str(params))
						
					log10_opt = \
						optimize.fmin_cobyla(minus_reduced_likelihood_function,
											 params,
											 constraints,
											 maxfun=3000,
											 #rhobeg=2.0,
											 rhoend=0.1,
											 iprint=0
											 )
					opt_minus_rlf = minus_reduced_likelihood_function(log10_opt)
					#print(opt_minus_rlf)
					log10_optimal_theta = theta_backToRealShape(log10_opt,self.theta.shape)
					
				except ValueError as ve:
					opt_minus_rlf = 999999999.
					k2 += 1
					raise ve
					print('Warning, exception raised in Cobyla')
				
				if(opt_minus_rlf != 999999999. ):
				
					optimal_theta = 10. ** log10_optimal_theta
					if self.verbose:
						print(optimal_theta)
					optimal_rlf_value, optimal_par = self.reduced_likelihood_function(theta=optimal_theta)

					# Compare the new optimizer to the best previous one
					if k > 0:
						if optimal_rlf_value > best_optimal_rlf_value:
							best_optimal_rlf_value = optimal_rlf_value
							best_optimal_par = optimal_par
							best_optimal_theta = optimal_theta
					else:
						best_optimal_rlf_value = optimal_rlf_value
						best_optimal_par = optimal_par
						best_optimal_theta = optimal_theta
					if self.verbose and self.random_start > 1:
						if (20 * k) / self.random_start > percent_completed:
							percent_completed = (20 * k) / self.random_start
							print("%s completed" % (5 * percent_completed))
					
					k += 1
					
				else:
					k2 += 1
					if(k2 == 50 and k==0):
						print("Cobyla Optimization failed. Try increasing the ``nugget``")
						best_optimal_theta = self.theta
						best_optimal_rlf_value, best_optimal_par = self.reduced_likelihood_function(theta=best_optimal_theta)
									
			optimal_rlf_value = best_optimal_rlf_value
			optimal_par = best_optimal_par
			optimal_theta = best_optimal_theta
			
		else:
			raise NotImplementedError("This optimizer ('%s') is not "
									  "implemented yet. Please contribute!"
									  % self.optimizer)

		return optimal_theta, optimal_rlf_value, optimal_par


	def _check_params(self, n_samples=None):

		# Check regression model
		if not callable(self.regr):
			if self.regr in self._regression_types:
				self.regr = self._regression_types[self.regr]
			else:
				raise ValueError("regr should be one of %s or callable, "
								 "%s was given."
								 % (self._regression_types.keys(), self.regr))

		# Check correlation model
		if not callable(self.corr):
			if self.corr in self._correlation_types:
				self.corr = self._correlation_types[self.corr]
			else:
				raise ValueError("corr should be one of %s or callable, "
								 "%s was given."
								 % (self._correlation_types.keys(), self.corr))


		# Force verbose type to bool
		self.verbose = bool(self.verbose)

		# Force normalize type to bool
		self.normalize = bool(self.normalize)

		# Check nugget value
		self.nugget = np.asarray(self.nugget)
		if np.any(self.nugget) < 0.:
			raise ValueError("nugget must be positive or zero.")
		if (n_samples is not None
				and self.nugget.shape not in [(), (n_samples,)]):
			raise ValueError("nugget must be either a scalar "
							 "or array of length n_samples.")


		# Force random_start type to int
		self.random_start = int(self.random_start)
