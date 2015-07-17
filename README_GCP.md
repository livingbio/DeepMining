### The Gaussian Copula Process class. ###

#### Parameters ####

| Name | Description |
|-------|-----------|
| regr | string or callable, optional
| |	A regression function returning an array of outputs of the linear regression functional basis. The number of observations n_samples should be greater than the size p of this basis. Default assumes a simple constant regression trend. Available built-in regression models are: 'constant', 'linear', 'quadratic' |
|corr | string or callable, optional |
| | A stationary autocorrelation function returning the autocorrelation between two points x and x'. Default assumes a squared-exponential autocorrelation model. Built-in correlation models are: 'squared_exponential', 'exponential_periodic' |
|theta | double array_like, optional |
| |	An array with shape (n_features, ) or (1, ). The parameters in the autocorrelation model. theta is the starting point for the maximum likelihood estimation of the best set of parameters. Default assumes isotropic autocorrelation model with theta0 = 1e-1. |
|thetaL | double array_like, optional |
| |	An array with shape matching theta0's. Lower bound on the autocorrelation parameters for maximum likelihood estimation. |
| thetaU | double array_like, optional |
| | An array with shape matching theta0's. Upper bound on the autocorrelation parameters for maximum likelihood estimation. |
|beta0 | double array_like, optional |
| |	The regression weight vector to perform Ordinary Kriging (OK). Default assumes Universal Kriging (UK) so that the vector beta of regression weights is estimated using the maximum likelihood principle. |
|try_optimize | boolean, optional |
| |	If True, perform maximum likelihood estimation to set the value of theta. Default is True. |
|normalize |  boolean, optional| 
| | Input X and observations y are centered and reduced wrt means and standard deviations estimated from the n_samples observations provided. Default is normalize = True so that data is normalized to ease maximum likelihood estimation. | 
| reNormalizeY |  boolean, optional| 
| | 	Normalize the warped Y values, ie. the values mapping(Yt), before fitting a GP. Default is False. | 
| n_clusters |  int, optional | 
| | 	If n_clusters > 1, a latent model is built by clustering the data with K-Means, into n_clusters clusters. Default is 1. | 
| coef_latent_mapping |  float, optional| 
| | 	If n_clusters > 1, this coefficient is used to interpolate the mapping function on the whole space from the mapping functions learned on each cluster. This acts as a smoothing parameter : if coef_latent_mapping == 0., each cluster contributes equally, and the greater it is the fewer mapping(x,y) takes into account the clusters in which x is not.	Default is 0.5.| 
| mapWithNoise |  boolean, optional| 
| | 	If True and if Y outputs contain multiple noisy observations for the same	x inputs, then all the noisy observations are used to compute Y's distribution	and learn the mapping function.	Otherwise, only the mean of the outputs, for a given input x, is considered. 	Default is False. | 
| useAllNoisyY | boolean, optional| 
| | 	If True and if Y outputs contain multiple noisy observations for the same	x inputs, then all the warped noisy observations are used to fit the GP.	Otherwise, only the mean of the outputs, for a given input x, is considered.	Default is False.| 
| model_noise |  string, optional| 
| | 	Method to model the noise.	If not None and if Y outputs contain multiple noisy observations for the same	x inputs, then the nugget (see below) is estimated from the standard	deviation of the multiple outputs for a given input x. Precisely the nugget	is multiplied by 100 * std (as data is usually normed and noise is usually	of the order of 1%).	Default is None, methods currently available are : 'EGN' (Estimated Gaussian Noise) | 
| nugget |  double or ndarray, optional | 
| | 	Introduce a nugget effect to allow smooth predictions from noisy data.  If nugget is an ndarray, it must be the same length as the	number of data points used for the fit.	The nugget is added to the diagonal of the assumed training covariance;	in this way it acts as a Tikhonov regularization in the problem.  In	the special case of the squared exponential correlation function, the	nugget mathematically represents the variance of the input values.	Default assumes a nugget close to machine precision for the sake of	robustness (nugget = 10. * MACHINE_EPSILON). | 
| random_start |  int, optional | 
| | 	The number of times the Maximum Likelihood Estimation should be	performed from a random starting point.	The first MLE always uses the specified starting point (theta0),	the next starting points are picked at random according to an	exponential distribution (log-uniform on [thetaL, thetaU]).	Default is 5.| 
| random_state|  integer or numpy.RandomState, optional| 
| | 	The generator used to shuffle the sequence of coordinates of theta in	the Welch optimizer. If an integer is given, it fixes the seed.	Defaults to the global numpy random number generator.| 
|verbose | boolean, optional |
| | A boolean specifying the verbose level. Default is verbose = False. |

#### Attributes ####
| Name | Description |
|-------|-----------|
| theta | array |
| | Specified theta OR the best set of autocorrelation parameters (the sought maximizer of the reduced likelihood function). |
|reduced_likelihood_function_value_ | array |
| | The optimal reduced likelihood function value. |
| centroids | array of shape (n_clusters,n_features) |
| | If n_clusters > 1, the array of the clusters' centroid. |
| density_functions | list of callable |
| | List of length n_clusters, containing the density estimations of the outputs on each cluster. |
| mapping | callable |
| | The mapping function such that mapping(Yt) has a gaussian \ distribution, where Yt is the output. The mapping is learned based on the KDE estimation of Yt's distribution |
| mapping_inv | callable |
| | The inverse mapping numerically computed by binomial search, such that mapping_inv[mapping(.)] == mapping[mapping_inv(.)] == id |
| mapping_derivative | callable |
| | The derivative of the mapping function. |


#### Notes  ####
This code is based on Scikit-learn's GP implementation.

#### References ####
On Gaussian processes :
* `H.B. Nielsen, S.N. Lophaven, H. B. Nielsen and J. Sondergaard.  DACE - A MATLAB Kriging Toolbox.` (2002) http://www2.imm.dtu.dk/~hbn/dace/dace.pdf
* `W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell, and M.D.  Morris (1992). Screening, predicting, and computer experiments.  Technometrics, 34(1) 15--25.` http://www.jstor.org/pss/1269548

On Gaussian Copula processes :
* `Wilson, A. and Ghahramani, Z. Copula processes. In Advances in NIPS 23, pp. 2460-2468, 2010 `
