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
import sampling_utils as utils 
from gcp import GaussianCopulaProcess

save_plots = False

### Set parameters ###
nugget = 1.e-10
all_n_clusters = [1,2,3]
corr_kernel = 'exponential_periodic'
GCP_mapWithNoise= False
sampling_model = 'GCP'
integratedPrediction = False
coef_latent_mapping = 0.1
prediction_size = 1000

### Set parameters ###
parameter_bounds = np.asarray( [[0,400]] )
training_size = 40


def scoring_function(x):
    return (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.

abs = np.atleast_2d(range(0,400)).T
f_plot = [scoring_function(i) for i in abs[:,0]]

x_training = []
y_training = []
for i in range(training_size):
	x = np.random.uniform(0,400)
	x_training.append(x)
	y_training.append(scoring_function(x))
x_training = np.atleast_2d(x_training).T

candidates = []
real_y = []
for i in range(prediction_size):
	x = [np.random.uniform(0,400)]
	candidates.append(x)
	real_y.append(scoring_function(x[0]))
real_y = np.asarray(real_y)
candidates = np.asarray(candidates)

count = -1
fig = plt.figure()

for n_clusters in all_n_clusters:

	count += 2
	ax = fig.add_subplot(len(all_n_clusters),2,count)
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
	print 'MSE', pred_error / (np.std(real_y) **2.)

	idx = np.argsort(candidates[:,0])
	s_candidates = candidates[idx,0]
	s_boundL = boundL[idx]
	s_boundU = boundU[idx]

	pred,MSE_bis = gcp.predict(np.atleast_2d(s_candidates).T,eval_MSE=True,transformY=False,eval_confidence_bounds=False,coef_bound = 1.96)
	gp_boundL = pred - 1.96*np.sqrt(MSE_bis)
	gp_boundU = pred + 1.96*np.sqrt(MSE_bis)
	t_f_plot =  [gcp.mapping(abs[i],f_plot[i],normalize=True) for i in range(len(f_plot))]
	t_y_training =  [gcp.mapping(x_training[i],y_training[i],normalize=True) for i in range(len(y_training))]

	if(save_plots):
		save_data = np.asarray([s_candidates,boundL,boundU,predictions,f_plot]).T
		np.savetxt('data_plot.csv',save_data,delimiter=',')

	ax.plot(abs,f_plot)
	l1, = ax.plot(candidates,predictions,'r+',label='GCP predictions')
	l3, = ax.plot(x_training,y_training,'bo',label='Training points')
	ax.fill(np.concatenate([s_candidates,s_candidates[::-1]]),np.concatenate([s_boundL,s_boundU[::-1]]),alpha=.5, fc='c', ec='None')


	ax = fig.add_subplot(len(all_n_clusters),2,count+1)
	ax.set_title('GP space')
	ax.plot(abs,t_f_plot)
	ax.plot(s_candidates,pred,'r+',label='GCP predictions')
	ax.plot(x_training,t_y_training,'bo',label='Training points')
	ax.fill(np.concatenate([s_candidates,s_candidates[::-1]]),np.concatenate([gp_boundL,gp_boundU[::-1]]),alpha=.5, fc='c', ec='None')

	if(save_plots):
		t_save_data = np.asarray([s_candidates,gp_boundL,gp_boundU,pred,np.asarray(t_f_plot)[:,0]]).T
		np.savetxt('gpspace_data_plot.csv',t_save_data,delimiter=',')
		training_points = np.asarray([x_training[:,0],y_training,np.asarray(t_y_training)[:,0]]).T
		np.savetxt('train_data_plot.csv',training_points,delimiter=',')

plt.legend()
plt.show()