import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats
import os

sys.path.append("../../")
from smart_sampling import smartSampling
from gcp import GaussianCopulaProcess

### Set parameters ###
parameter_bounds = np.asarray( [[0,400]] )
training_size = 30
nugget = 1e-10
n_clusters = 1
corr_kernel = 'squared_exponential'
integratedPrediction = True

def scoring_function(x):
    return (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.

x_training = []
y_training = []
for i in range(training_size):
	x = np.random.uniform(parameter_bounds[0][0],parameter_bounds[0][1])
	x_training.append(x)
	y_training.append(scoring_function(x))
x_training = np.atleast_2d(x_training).T


gcp = GaussianCopulaProcess(nugget = nugget,
                            corr=corr_kernel,
                            random_start=5,
                            normalize = True,
                            n_clusters=n_clusters)
gcp.fit(x_training,y_training)
print 'GCP fitted'
print 'Theta', gcp.theta
if(n_clusters > 1):
	centers = np.asarray([gcp.centroids[i][0]* gcp.X_std + gcp.X_mean for i in range(n_clusters) ], dtype=np.int32)

m = np.mean(y_training)
s = np.std(y_training)
y_mean, y_std = gcp.raw_y_mean,gcp.raw_y_std
x_density_plot = (np.asarray ( range(np.int(m *100.- 100.*(s)*10.),np.int(m*100. + 100.*(s)*10.)) ) / 100. - y_mean)/ y_std

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
for i in range(n_clusters):
	plt_density_gcp = gcp.density_functions[i](x_density_plot)
	if(n_clusters > 1):
		l = 'Cluster ' + str(centers[i])
	else:
		l = 'KDE estimation'
	plt.plot(x_density_plot*y_std + y_mean,plt_density_gcp,label=l)
plt.legend()
ax.set_title('Density estimation')

candidates = np.atleast_2d(range(80)).T * 5
#simple_prediction = gcp.predict(candidates,integratedPrediction=False)
prediction = gcp.predict(candidates,integratedPrediction=True)

# plot results
abs = range(0,400)
f_plot = [scoring_function(i) for i in abs]
ax = fig.add_subplot(1,2,2)
plt.plot(abs,f_plot)
plt.plot(x_training,y_training,'bo')
#plt.plot(candidates,simple_prediction,'go',label='Simple prediction')
plt.plot(candidates,prediction,'r+',label='Integrated prediction')
plt.legend()
ax.set_title('GCP estimation')
plt.show()