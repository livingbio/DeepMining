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
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess

### Set parameters ###
parameter_bounds = np.asarray( [[0,400]] )
training_size = 50
nugget = 1.e-10
n_clusters_max = 3
corr_kernel = 'exponential_periodic' # 'squared_exponential'

def scoring_function(x):
    return (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.

x_training = []
y_training = []
for i in range(training_size):
	#x = np.random.uniform(parameter_bounds[0][0],parameter_bounds[0][1])
	x = np.random.uniform(0,400)
	x_training.append(x)
	y_training.append(scoring_function(x))
x_training = np.atleast_2d(x_training).T


fig1 = plt.figure()
plt.title('Density estimation')
fig2 = plt.figure()
plt.title("GCP prediction")

n_rows = (n_clusters_max+1)/2
if not((n_clusters_max+1)% 2 == 0):
	n_rows += 1

mf_plt_axis = np.asarray(range(100)) / 100.
mapping_functions_plot = []

for n_clusters in range(1,n_clusters_max+1):

	gcp = GaussianCopulaProcess(nugget = nugget,
	                            corr=corr_kernel,
	                            random_start=5,
	                            normalize = True,
	                            coef_latent_mapping = 0.4,
	                            n_clusters=n_clusters)
	gcp.fit(x_training,y_training)
	likelihood = gcp.reduced_likelihood_function_value_
	print 'LGCP-'+str(n_clusters)+' fitted'
	print 'Theta', gcp.theta
	print 'Likelihood',likelihood

	if(n_clusters > 1):
		centers = np.asarray([gcp.centroids[i][0]* gcp.X_std + gcp.X_mean for i in range(gcp.n_clusters) ], dtype=np.int32)

	m = np.mean(y_training)
	s = np.std(y_training)
	y_mean, y_std = gcp.raw_y_mean,gcp.raw_y_std
	x_density_plot = (np.asarray ( range(np.int(m *100.- 100.*(s)*10.),np.int(m*100. + 100.*(s)*10.)) ) / 100. - y_mean)/ y_std


	ax1 = fig1.add_subplot(n_rows,2,n_clusters)
	for i in range(gcp.n_clusters):
		plt_density_gcp = gcp.density_functions[i](x_density_plot)
		if(n_clusters > 1):
			l = 'Cluster ' + str(centers[i])
		else:
			l = 'KDE estimation'
		ax1.plot(x_density_plot*y_std + y_mean,plt_density_gcp,label=l)
	ax1.legend()
	ax1.set_title('n_clusters == ' + str(n_clusters) )

	candidates = np.atleast_2d(range(80)).T * 5
	prediction = gcp.predict(candidates)

	# plot results
	abs = range(0,400)
	f_plot = [scoring_function(i) for i in abs]
	ax2 = fig2.add_subplot(n_rows,2,n_clusters)
	plt.plot(abs,f_plot)
	plt.plot(x_training,y_training,'bo')
	#plt.plot(candidates,simple_prediction,'go',label='Simple prediction')
	plt.plot(candidates,prediction,'r+',label='n_clusters == ' + str(n_clusters))
	plt.axis([0,400,0,1.])
	ax2.set_title('Likelihood = ' + str(likelihood))
	plt.legend()

	mapping_functions_plot.append( [gcp.mapping(200,mf_plt_axis[i],normalize=True) for i in range(100)])
## with GP ##

gp = GaussianProcess(theta0=.1 ,
				 thetaL = 0.001,
				 thetaU = 10.,
				 random_start = 5,
				 nugget=nugget)
gp.fit(x_training,y_training)
likelihood = gp.reduced_likelihood_function_value_
print 'GP'
print 'Theta',gp.theta_
print 'Likelihood',likelihood
candidates = np.atleast_2d(range(80)).T * 5
prediction = gp.predict(candidates)

# plot results
abs = range(0,400)
ax2 = fig2.add_subplot(n_rows,2,n_clusters_max+1)
plt.plot(abs,f_plot)
plt.plot(x_training,y_training,'bo')
plt.plot(candidates,prediction,'r+',label='GP')
plt.axis([0,400,0,1.])
ax2.set_title('Likelihood = ' + str(likelihood))
plt.legend()

ax1 = fig1.add_subplot(n_rows,2,n_clusters_max+1)
for i in range(n_clusters_max):
	ax1.plot(mf_plt_axis,np.asarray(mapping_functions_plot[i])[:,0],label=str(i+1)+' clusters')
ax1.set_title('Mapping functions')
ax1.legend(loc=4)

plt.show()