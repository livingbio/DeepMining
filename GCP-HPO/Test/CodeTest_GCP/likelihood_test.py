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
import sys

sys.path.append("../../")
sys.path.append("../Branin/")
sys.path.append("../Hartmann/")
from branin import branin
from har6 import har6
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess

### Set parameters ###
t_size = [20,50,80]
nugget = 1.e-10
n_clusters_max = 4
integratedPrediction = False
n_tests = 20
log_likelihood = False
print 'Average on n_tests = 20, log_likelihood = False, nugget = 1.e-10, integratedPrediction = False'

def artificial_f(x):
	x = x[0]
	res = (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.
	return [res]

def branin_f(p_vector):
	x,y = p_vector
	x = x -5.
	y= y
	return branin(x,y)

all_parameter_bounds = {'artificial_f': np.asarray( [[0,400]] ),
						'branin': np.asarray( [[0,15],[0,15]] ),
						'har6' : np.asarray( [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]) }

tests = {'function' : ['artificial_f','branin','har6'],
		'corr_kernel': ['squared_exponential','exponential_periodic'] }

functions = {'artificial_f':artificial_f,'branin':branin_f,'har6':har6}


def run_test(training_size,scoring_function,parameter_bounds,corr_kernel,n_cluster,prior='GCP',log=True):

	x_training = []
	y_training = []
	for i in range(training_size):
		x = [np.random.uniform(parameter_bounds[j][0],parameter_bounds[j][1]) for j in range(parameter_bounds.shape[0])]
		x_training.append(x)
		y_training.append(scoring_function(x)[0])

	if(prior == 'GP'):
		gp = GaussianProcess(theta0=.1 *np.ones(parameter_bounds.shape[0]),
						 thetaL = 0.001 * np.ones(parameter_bounds.shape[0]),
						 thetaU = 10. * np.ones(parameter_bounds.shape[0]),
						 random_start = 5,
						 nugget=nugget)
		gp.fit(x_training,y_training)
		likelihood = gp.reduced_likelihood_function_value_
	else:
		gcp = GaussianCopulaProcess(nugget = nugget,
		                            corr=corr_kernel,
		                            random_start=5,
		                            normalize = True,
		                            coef_latent_mapping = 0.4,
		                            n_clusters=n_clusters)
		gcp.fit(x_training,y_training)
		likelihood = gcp.reduced_likelihood_function_value_

	if not log:
		likelihood = np.exp(likelihood)

	return likelihood


for s in t_size:
	print 'Training size :',s
	for f in tests['function']:
		print ' **  Test function',f,' ** '
		scoring_function = functions[f]
		parameter_bounds = all_parameter_bounds[f]
		likelihood = [run_test(s,scoring_function,parameter_bounds,'',0,prior='GP',log=log_likelihood) for j in range(n_tests)]
		print '\t\t\t\t > GP - Likelihood =',np.mean(likelihood),'\t',np.std(likelihood),'\n'
		for k in tests['corr_kernel']:
			for n_clusters in range(1,n_clusters_max):
				likelihood = [run_test(s,scoring_function,parameter_bounds,k,n_clusters,log=log_likelihood) for j in range(n_tests)]
				print 'corr_kernel:',k,'- n_clusters:',n_clusters,'\tLikelihood =',np.mean(likelihood),'\t',np.std(likelihood)
			print ''
		print ''
	print ''


