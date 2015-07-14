import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../../")
from smart_sampling import smartSampling
from gcp import GaussianCopulaProcess

### Set parameters ###
parameter_bounds = np.asarray( [[0,400]] )
nugget = 1e-10
n_clusters = 1
cluster_evol ='constant'
corr_kernel = 'squared_exponential'
GCPconsiderAllObs= False
model_noise = None
sampling_model = 'GCP'
n_candidates= 100
n_random_init= 10
nb_GCP_steps = 10
nb_iter_final = 0
acquisition_function = 'MaxUpperBound'

def scoring_function(x):
    res = (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.
    return [res]


X,Y = smartSampling(nb_GCP_steps,parameter_bounds,scoring_function,
											  isInt=True,
							                  corr_kernel = corr_kernel,
							                  acquisition_function = acquisition_function,
							                  GCPconsiderAllObs1=GCPconsiderAllObs,
							          		  model_noise = model_noise,
							                  model = sampling_model, 
							                  n_candidates=n_candidates,
							                  n_random_init=n_random_init,
							                  n_final_iter=nb_iter_final,
							                  n_clusters=1, 
							                  cluster_evol = cluster_evol,
							                  verbose=True,
							                  detailed_res = False)

X = np.asarray(X[0])
Y = np.asarray(Y[0])

X_init = X[:n_random_init]
Y_init = Y[:n_random_init]

# plot results
abs = range(0,400)
f_plot = [scoring_function(i) for i in abs]
fig = plt.figure()
plt.plot(abs,f_plot)
plt.plot(X,Y,'ro',label='GCP query points')
plt.plot(X_init,Y_init,'bo',label='Random initialization')
plt.title('Smart sampling process')
plt.legend()
plt.show()