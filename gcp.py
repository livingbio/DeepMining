# -*- coding: utf-8 -*-

# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

from __future__ import print_function


import numpy as np
from scipy import linalg, optimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.gaussian_process import regression_models as regression
from scipy.stats import norm
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial import distance
from GCP_utils import *
from sklearn_utils import *

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
		Available built-in regression models are::

			'constant', 'linear', 'quadratic'

	mapping : the mapping function such that mapping(Yt) is a gaussian process, where Yt
		is the output
		Default is log.
		
	mapping_inv : the inverse mapping, such that mapping_inv[mapping(.)] == mapping[mapping_inv(.)] == id
		Default is exp.
		
	corr : string or callable, optional
		A stationary autocorrelation function returning the autocorrelation
		between two points x and x'.
		Default assumes a squared-exponential autocorrelation model.
		Built-in correlation models are::

			'absolute_exponential', 'squared_exponential',
			'generalized_exponential', 'cubic', 'linear'

	beta0 : double array_like, optional
		The regression weight vector to perform Ordinary Kriging (OK).
		Default assumes Universal Kriging (UK) so that the vector beta of
		regression weights is estimated using the maximum likelihood
		principle.

	storage_mode : string, optional
		A string specifying whether the Cholesky decomposition of the
		correlation matrix should be stored in the class (storage_mode =
		'full') or not (storage_mode = 'light').
		Default assumes storage_mode = 'full', so that the
		Cholesky decomposition of the correlation matrix is stored.
		This might be a useful parameter when one is not interested in the
		MSE and only plan to estimate the BLUP, for which the correlation
		matrix is not required.

	verbose : boolean, optional
		A boolean specifying the verbose level.
		Default is verbose = False.

	theta0 : double array_like, optional
		An array with shape (n_features, ) or (1, ).
		The parameters in the autocorrelation model.
		If thetaL and thetaU are also specified, theta0 is considered as
		the starting point for the maximum likelihood estimation of the
		best set of parameters.
		Default assumes isotropic autocorrelation model with theta0 = 1e-1.

	thetaL : double array_like, optional
		An array with shape matching theta0's.
		Lower bound on the autocorrelation parameters for maximum
		likelihood estimation.
		Default is None, so that it skips maximum likelihood estimation and
		it uses theta0.

	thetaU : double array_like, optional
		An array with shape matching theta0's.
		Upper bound on the autocorrelation parameters for maximum
		likelihood estimation.
		Default is None, so that it skips maximum likelihood estimation and
		it uses theta0.

	normalize : boolean, optional
		Input X and observations y are centered and reduced wrt
		means and standard deviations estimated from the n_samples
		observations provided.
		Default is normalize = True so that data is normalized to ease
		maximum likelihood estimation.

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

	optimizer : string, optional
		A string specifying the optimization algorithm to be used.
		Default uses 'fmin_cobyla' algorithm from scipy.optimize.
		Available optimizers are::

			'fmin_cobyla', 'Welch'

		'Welch' optimizer is dued to Welch et al., see reference [WBSWM1992]_.
		It consists in iterating over several one-dimensional optimizations
		instead of running one single multi-dimensional optimization.

	random_start : int, optional
		The number of times the Maximum Likelihood Estimation should be
		performed from a random starting point.
		The first MLE always uses the specified starting point (theta0),
		the next starting points are picked at random according to an
		exponential distribution (log-uniform on [thetaL, thetaU]).
		Default does not use random starting point (random_start = 1).

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

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.gaussian_process import GaussianProcess
	>>> X = np.array([[1., 3., 5., 6., 7., 8.]]).T
	>>> y = (X * np.sin(X)).ravel()
	>>> gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
	>>> gp.fit(X, y)                                      # doctest: +ELLIPSIS
	GaussianProcess(beta0=None...
			...

	Notes
	-----
	The presentation implementation is based on a translation of the DACE
	Matlab toolbox, see reference [NLNS2002]_.

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
				 theta=np.asarray([40,30,20,.05,1.,1.,2.,.1]),
				 thetaL=np.asarray([10.,1.,1.,.01,1.,0.1,0.1,0.01]),
				 thetaU=np.asarray([90,50,20,10.,5.,10.,10.,1.]), 
				 try_optimize=True,
				 random_start=10, 
				 normalize=True,
				 x_wrapping='none',
				 n_clusters = 1,
				 nugget=10. * MACHINE_EPSILON,
				 random_state=None):
 
		self.regr = regr
		self.beta0 = None
		self.storage_mode = 'full'
		self.verbose = verbose
		self.theta = theta
		self.thetaL = thetaL
		self.thetaU = thetaU
		self.normalize = normalize
		self.nugget = nugget
		self.optimizer = 'fmin_cobyla'
		self.random_start = random_start
		self.random_state = random_state
		self.try_optimize = try_optimize
		self.n_clusters = n_clusters
		self.density_functions = None
		self.x_wrapping = x_wrapping
		if (corr == 'squared_exponential'):
			self.corr = sq_exponential
			self.theta = np.asarray([0.1])
			self.thetaL = np.asarray([0.001])
			self.thetaU = np.asarray([10.])
		else:
			self.corr = exponential_periodic
		
	def mapping(self,x,t):
		v = 0.
		if( t < 2047483647):
			if(self.n_clusters > 1):
				coefs = np.ones(self.n_clusters)
				val = np.zeros(self.n_clusters)
				for w in range(self.n_clusters):
					## coefficients are :
					#	 exp{  - sum [ (d_i /std_i) **2 ]  }
					coefs[w] =  np.exp(- np.sum( ((x -self.centroids[w])/self.clusters_std)**2. ) )
					temp  =  self.density_functions[w].integrate_box_1d(self.low_bound, t)
					temp = min(0.999999998,temp)
					val[w] = ( norm.ppf( temp) ) * coefs[w]
				s = np.sum(coefs)
				val = val / s
				v = np.sum(val)
			else:
				temp =  min(0.999999998,self.density_functions[0].integrate_box_1d(self.low_bound, t) )
				v = norm.ppf(temp)
		else:
			v = 6.3
			
		return [v]
		

	def mapping_inv(self,x,t):
		if(t< 6.15):
			def map(t):
				return self.mapping(x,t)
			lo, hi = find_bounds(map, t)
			res = binary_search(map, t, lo, hi)				
			return [res]
		else:
			return [2047483647]


	def init_mappings(self):
		# We assume y is one-dimensional
	
		# Perform KMeans and store results
		if(self.n_clusters > 1):
			kmeans = KMeans(n_clusters=self.n_clusters)
			windows_idx = kmeans.fit_predict(self.X)
			self.centroids = kmeans.cluster_centers_
			print ("Centroids")
			print (self.X_std*self.centroids + self.X_mean)
			
			# Compute the density function for each sub-window
			density_functions = []
			clusters_std = []
			for w in range(self.n_clusters):
				cluster_points_y_values = np.copy((self.raw_y[ windows_idx == w])[:,0])
				clusters_std.append( np.std( self.X[ windows_idx == w], axis=0) ) ### this is a (Xdim) array
				print('cluster '+str(w)+' size ' + str(cluster_points_y_values.shape))
				density_functions.append(stats.gaussian_kde(cluster_points_y_values) )
			density_functions = np.asarray( density_functions)
			self.density_functions = density_functions
			self.clusters_std = np.asarray(clusters_std)
		
		else:
			self.density_functions = np.asarray( [ stats.gaussian_kde(self.raw_y[:,0]) ])
		
		
	def update_copula_params(self):
		size = self.raw_y.shape[0]
		y = [ self.mapping(self.X[i],self.raw_y[i]) for i in range(size)]
		y = np.asarray(y)
		
		# Normalize data
		y_mean = np.mean(y, axis=0)
		y_std = np.std(y, axis=0)
		y_std[y_std == 0.] = 1.
		y = (y - y_mean) / y_std

		# Calculate matrix of distances D between samples
		D, ij = l1_cross_distances(self.X)
		if (np.min(np.sum(D, axis=1)) == 0.
				and self.corr != correlation.pure_nugget):
			raise Exception("Multiple input features cannot have the same"
							" target value.")

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
			
			
	def fit(self, X, y):
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

		Returns
		-------
		gcp : self
			A fitted Gaussian Copula Process model object awaiting data to perform
			predictions.
		"""
		# Run input checks
		self._check_params()
		X = array2d(X)
		y = np.asarray(y)
		self.y_ndim_ = y.ndim
		if y.ndim == 1:
			y = y[:, np.newaxis]
		else:
			print('Warning: code is not ready for y outputs with dimension > 1')
			
		# Reshape theta if it is one dimensional and X is not
		#print('theta shape '+str(self.theta.shape))
		if not (self.theta.ndim == X.ndim):
			if not(self.theta.ndim == 1):
				print('Warning : theta has not the right shape')
			self.theta = (np.ones((X.ndim,self.theta.shape[0])) * self.theta ).T
			self.thetaL = (np.ones((X.ndim,self.thetaL.shape[0])) * self.thetaL ).T
			self.thetaU = (np.ones((X.ndim,self.thetaU.shape[0])) * self.thetaU ).T
			#print('theta has new shape '+str(self.theta.shape))
		
			
		self.random_state = check_random_state(self.random_state)
		self.raw_y = y
		self.low_bound = np.min([-500., 5. * np.min(y)])
		
		X, y = check_arrays(X, y)

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
			# center and scale X if necessary
			X = (X - X_mean) / X_std
		else:
			X_mean = np.zeros(1)
			X_std = np.ones(1)

		# Set attributes
		self.X = X
		self.X_mean, self.X_std = X_mean, X_std

		if(self.x_wrapping != 'none'):
			X = GCP_Xwrapping(X,self.x_wrapping)
		
		# initialize mapping only if needed, i.e. it hasn't be done 
		# yet of if we want to optimize the GCP hyperparameters
		if (self.try_optimize or (self.density_functions is None)):
			self.init_mappings()
		
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

	def predict(self, X, eval_MSE=False, eval_confidence_bounds=False,upperBoundCoef=1.96, batch_size=None):
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

		batch_size : integer, optional
			An integer giving the maximum number of points that can be
			evaluated simultaneously (depending on the available memory).
			Default is None so that all given points are evaluated at the same
			time.

		Returns
		-------
		y : array_like, shape (n_samples, ) or (n_samples, n_targets)
			An array with shape (n_eval, ) if the Gaussian Process was trained
			on an array of shape (n_samples, ) or an array with shape
			(n_eval, n_targets) if the Gaussian Process was trained on an array
			of shape (n_samples, n_targets) with the Best Linear Unbiased
			Prediction at x.

		MSE : array_like, optional (if eval_MSE == True)
			An array with shape (n_eval, ) or (n_eval, n_targets) as with y,
			with the Mean Squared Error at x.
		"""

		# Check input shapes
		X = array2d(X)
		n_eval, _ = X.shape
		n_samples, n_features = self.X.shape
		n_samples_y, n_targets = self.y.shape

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

		if(self.x_wrapping != 'none'):
			X = GCP_Xwrapping(X,self.x_wrapping)
			
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
		real_y = [ self.mapping_inv(X[i],y[i][0]) for i in range(size)]
		real_y = np.asarray(real_y)
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
				
				if(eval_confidence_bounds):
					sigma = np.sqrt(MSE)
					warped_y_with_boundL = warped_y - 1.9600 * sigma
					warped_y_with_boundU = warped_y + upperBoundCoef * sigma
					pred_with_boundL = np.asarray( [ self.mapping_inv(X[i],warped_y_with_boundL[i])[0] for i in range(size) ] )
					pred_with_boundU = np.asarray( [ self.mapping_inv(X[i],warped_y_with_boundU[i])[0] for i in range(size)] )
					return y,MSE,pred_with_boundL,pred_with_boundU
						
				else:
					if(self.n_clusters > 1):
						center = np.mean(self.centroids)
					else:
						center = 0.
					sigma = np.sqrt(MSE)
					coefU = (self.mapping_inv(center,self.y_mean + 1.96*np.mean(sigma))[0] - self.mapping_inv(center,self.y_mean)[0])/1.96
					coefL = -(self.mapping_inv(center,self.y_mean - 1.96*np.mean(sigma))[0] - self.mapping_inv(center,self.y_mean)[0])/1.96
					return y, MSE, coefL, coefU
			
			else:
				return y, MSE

		else:
			return y


	def reduced_likelihood_function(self, theta=None,verb=False):
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
			if (np.min(np.sum(D, axis=1)) == 0.
					and self.corr != correlation.pure_nugget):
				raise Exception("Multiple X are not allowed")
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
					theta=10. ** x_reshaped,
					verb=False)[0]
						
			constraints = []
			# http://stackoverflow.com/questions/25985868/scipy-why-isnt-cobyla-respecting-constraint
			
			# Cobyla takes only one dimensional array
			conL = np.log10(theta_toOneDim(self.thetaL))
			conU = np.log10(theta_toOneDim(self.thetaU))
			
			def kernel_coef(x):
				return(100. - ((10. ** x[0]) + (10. ** x[1]) + (10. ** x[2]) ))
			
			for idx in range(self.theta.size-3):
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
					#raise ve
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
