import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../../")
from gcp import GaussianCopulaProcess

### Set parameters ###
parameter_bounds = np.asarray( [[0,400]] )
training_size = 20
nugget = 1.e-10
n_clusters_max = 4
corr_kernel = 'squared_exponential'
integratedPrediction = False

def scoring_function(x):
    return (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.

x_training = []
y_training = []
for i in range(training_size):
	x = np.random.uniform(parameter_bounds[0][0],parameter_bounds[0][1])
	x_training.append(x)
	y_training.append(scoring_function(x))
x_training = np.atleast_2d(x_training).T


fig1 = plt.figure()
plt.title('Density estimation')
fig2 = plt.figure()
plt.title("GCP prediction")

n_rows = n_clusters_max/2
if not(n_clusters_max% 2 == 0):
	n_rows += 1

for n_clusters in range(1,n_clusters_max+1):

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


	ax1 = fig1.add_subplot(n_rows,2,n_clusters)
	for i in range(n_clusters):
		plt_density_gcp = gcp.density_functions[i](x_density_plot)
		if(n_clusters > 1):
			l = 'Cluster ' + str(centers[i])
		else:
			l = 'KDE estimation'
		ax1.plot(x_density_plot*y_std + y_mean,plt_density_gcp,label=l)
	ax1.legend()
	ax1.set_title('n_clusters == ' + str(n_clusters))

	candidates = np.atleast_2d(range(80)).T * 5
	#simple_prediction = gcp.predict(candidates,integratedPrediction=False)
	prediction = gcp.predict(candidates,integratedPrediction=integratedPrediction)

	# plot results
	abs = range(0,400)
	f_plot = [scoring_function(i) for i in abs]
	ax2 = fig2.add_subplot(n_rows,2,n_clusters)
	plt.plot(abs,f_plot)
	plt.plot(x_training,y_training,'bo')
	#plt.plot(candidates,simple_prediction,'go',label='Simple prediction')
	plt.plot(candidates,prediction,'r+',label='Integrated prediction')
	ax2.set_title('n_clusters == ' + str(n_clusters))
	plt.legend()

plt.show()