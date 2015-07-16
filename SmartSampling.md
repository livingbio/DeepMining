#### Parameters ####

| Name | Description|
|------|------------|
| n_iter | int|
||Number of smart iterations to perform.|
|parameters_bounds | ndarray
||The bounds between which to sample the parameters. parameter_bounds.shape = [n_parameters,2], parameter_bounds[i] = [ lower bound for parameter i, upper bound for parameter i]|
|score_function | callable |
||A function that computes the output, given some parameters. This is the function to optimize. /!\ Always put data_size as the first parameter, if not None |
|model | string, optional |
||The model to run. Choose between : GCP (runs only the Gaussian Copula Process), GP (runs only the Gaussian Process), random (samples at random), GCPR (GCP and random), all (runs all models) |
|acquisition function | string, optional |
||Function to maximize in order to choose the next parameter to test. Simple : maximize the predicted output; MaxUpperBound : maximize the upper confidence bound; MaxLowerBound : maximize the lower confidence bound; EI : maximizes the expected improvement. /!\ EI is not available for GP. Default is 'MaxUpperBound' |
|corr_kernel | string, optional|
||Correlation kernel to choose for the GCP. Possible choices are : exponential_periodic (a linear combination of 3 classic kernels); squared_exponential. Default is 'exponential_periodic'. |
| n_random_init | int, optional|
||Number of random iterations to perform before the smart sampling. Default is 30.|
|n_candidates | int, optional|
||Number of random candidates to sample for each GCP / GP iterations Default is 2000. |
|n_clusters |int, optional|
||Number of clusters used in the parameter space to build a variable mapping for the GCP. Default is 1. |
|cluster_evol | string {'constant', 'step', 'variable'}, optional |
||Method used to set the number of clusters. If 'constant', the number of clusters is set with n_clusters. If 'step', start with one cluster, and set n_clusters after 20 smart steps. If 'variable', start with one cluster and increase n_clusters by one every 30 smart steps. Default is constant. |
|n_clusters_max | int, optional|
||The maximum value for n_clusters. Default is 5. |
|nugget | float, optional |
||The nugget to set for the Gaussian Copula Process or Gaussian Process. Default is 1.e-7.|
|GCP_mapWithNoise | boolean, optional|
||If True and if Y outputs contain multiple noisy observations for the same x inputs, then all the noisy observations are used to compute Y's distribution and learn the mapping function. Otherwise, only the mean of the outputs, for a given input x, is considered. Default is False. |
|GCP_useAllNoisyY | boolean, optional |
||If True and if Y outputs contain multiple noisy observations for the same x inputs, then all the warped noisy observations are used to fit the GP. Otherwise, only the mean of the outputs, for a given input x, is considered. Default is False. |
|model_noise | string {'EGN',None}, optional |
||Method to model the noise. If not None and if Y outputs contain multiple noisy observations for the same x inputs, then the nugget is estimated from the standard deviation of the multiple outputs for a given input x. Default is None.|
| isInt | boolean or (n_parameters) numpy array |
|| Specify which parameters are integers. If isInt is a boolean, all parameters are assumed to have the same type. It is better to fix isInt=True rather than converting floating parameters as integers in the scoring function, because this would generate a discontinuous scoring function (whereas GP / GCP assume that the function is smooth). |
|detailed_res | boolean, optional|
||Specify if the method should return only the parameters and mean outputs or all the details, see below.|


#### Returns ####
| Name | Description|
|------|------------|
| all_parameters |the parameters tested during the process. |
||	A list of length the number of models to use.	For each model, this contains a ndarray of size (n_parameters_tested,n_features).|
|all_raw_outputs | the detailed observations (if detailed_res == True ).|
||	A list of length the number of models to use.	For each model, this contains a list of length the number of parameters tested during the process,	for each parameter, the entry is a list containing all the (noisy) observations. |
|all_mean_outputs | the mean values of the outputs.|
||	A list of length the number of models to use. For each model, this contains a list of length the number of parameters tested during the process, and the values correspond to the mean of the (noisy) observations.|
|all_std_outputs | the standard deviation of the observations (if detailed_res == True ).|
||	A list of length the number of models to use.	For each model, this contains a list of length the number of parameters tested during the process,	for each parameter, the entry is the standard deviation of the (noisy) observations.|
|all_param_path | the path of the tested parameters.|
||	ndarray of size (n_models, n_total_iter,n_features), where n_total_iter is the sum of	n_random_init, n_iter and n_final_iter.	For each model, the ndarray stores the parameters tested and the order. The main difference	between all_parameters is that all_parameters cannot contain twice the same parameter, but	all_param_path can.|
