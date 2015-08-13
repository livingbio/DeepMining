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
import matplotlib.pyplot as plt

sys.path.append("../../")
sys.path.append("../Branin/")
from branin import branin
import sampling_utils as utils 
from gcp import GaussianCopulaProcess
from mpl_toolkits.mplot3d import Axes3D

save_plots = False

### Set parameters ###
nugget = 1.e-10
all_n_clusters = [1]
corr_kernel = 'squared_exponential'
GCP_mapWithNoise= False
sampling_model = 'GCP'
integratedPrediction = False
coef_latent_mapping = 0.1
prediction_size = 1000

### Set parameters ###
parameter_bounds = np.asarray( [[0,15],[0,15]] )
training_size = 50

def branin_f(p_vector):
	x,y = p_vector
	x = x -5.
	y= y
	return branin(x,y)

x_training = []
y_training = []
for i in range(training_size):
	x = [np.random.uniform(parameter_bounds[j][0],parameter_bounds[j][1]) for j in range(parameter_bounds.shape[0])]
	x_training.append(x)
	y_training.append(branin_f(x)[0])
x_training = np.asarray(x_training)

candidates = []
real_y = []
for i in range(prediction_size):
	x = [np.random.uniform(parameter_bounds[j][0],parameter_bounds[j][1]) for j in range(parameter_bounds.shape[0])]
	candidates.append(x)
	real_y.append(branin_f(x)[0])
real_y = np.asarray(real_y)
candidates = np.asarray(candidates)

for n_clusters in all_n_clusters:

	fig = plt.figure()
	ax = fig.add_subplot(1,2,1, projection='3d')
	ax.set_title("GCP prediction")

	gcp = GaussianCopulaProcess(nugget = nugget,
								corr = corr_kernel,
								random_start = 5,
								n_clusters = n_clusters,
	                            coef_latent_mapping = coef_latent_mapping,
							 	mapWithNoise = GCP_mapWithNoise,
				 				useAllNoisyY = False,
				 				model_noise = None,
								try_optimize = True)
	gcp.fit(x_training,y_training)

	print '\nGCP fitted'
	print 'Theta', gcp.theta
	print 'Likelihood', np.exp(gcp.reduced_likelihood_function_value_)

	predictions,MSE,boundL,boundU = \
						gcp.predict(candidates,eval_MSE=True,eval_confidence_bounds=True,coef_bound = 1.96,integratedPrediction=integratedPrediction)

	pred_error = np.mean( (predictions - np.asarray(real_y) ) **2. )
	print 'MSE', pred_error
	print 'Normalized error', np.sqrt(pred_error) /np.std(real_y)
	 
	pred,MSE_bis = gcp.predict(candidates,eval_MSE=True,transformY=False,eval_confidence_bounds=False,coef_bound = 1.96)

	t_f_plot =  [gcp.mapping(candidates[i],real_y[i],normalize=True) for i in range(real_y.shape[0])]
	t_y_training =  [gcp.mapping(x_training[i],y_training[i],normalize=True) for i in range(len(y_training))]

	ax.scatter(x_training[:,0],x_training[:,1],y_training,c='g',label='Training points',alpha=0.5)
	ax.scatter(candidates[:,0],candidates[:,1],real_y,c='b',label='Branin function',alpha=0.5)
	ax.scatter(candidates[:,0],candidates[:,1],predictions,c='r',label='predictions',marker='+')

	ax = fig.add_subplot(1,2,2, projection='3d')
	ax.set_title('GP space')
	ax.scatter(x_training[:,0],x_training[:,1],t_y_training,c='g',label='Training points',alpha=0.5)
	ax.scatter(candidates[:,0],candidates[:,1],t_f_plot,c='b',label='Branin function',alpha=0.5)
	ax.scatter(candidates[:,0],candidates[:,1],pred,c='r',label='predictions',marker='+')

plt.legend()
plt.show()