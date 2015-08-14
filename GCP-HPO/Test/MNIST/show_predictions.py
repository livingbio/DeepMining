import numpy as np
import sys

sys.path.append("../../")
import sampling_utils as utils
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

# MNIST data
mnist_output = []
f =open(("../MNIST/scoring_function/output.csv"),'r')
for l in f:
  l = l[1:-3]
  string_l = l.split(',')
  mnist_output.append( np.mean([ float(i) for i in string_l] ))
f.close()
mnist_params = np.genfromtxt(("../MNIST/scoring_function/params.csv"),delimiter=',')
params_bis = (10*mnist_params[:,3] +  5* mnist_params[:,0] + mnist_params[:,1]) / 10.
params_bis2 = mnist_params[:,2] + (3.* mnist_params[:,4] / 10. )
t_params = np.asarray([params_bis,params_bis2]).T
mnist_KNN = NearestNeighbors()
mnist_KNN.fit(t_params)

parameter_bounds = np.asarray( [[0,2],[0,5],[5,31],[1,5],[0,4]] )

def get_exp_data(n_exp,length,isRand=False):
	n_model = 0
	exp_dir = 'exp_results'
	if(isRand):
	    o_dir = (exp_dir+"/rand/exp"+str(n_exp)+"/output_" +str(n_model)+".csv")
	    p_dir = (exp_dir+"/rand/exp"+str(n_exp)+"/param_" +str(n_model)+".csv")
	    path_dir = (exp_dir+"/rand/exp"+str(n_exp)+"/param_path_" +str(n_model)+".csv")
	else:
	    o_dir = (exp_dir+"/exp"+str(n_exp)+"/output_" +str(n_model)+".csv")
	    p_dir = (exp_dir+"/exp"+str(n_exp)+"/param_" +str(n_model)+".csv")
	    path_dir = (exp_dir+"/exp"+str(n_exp)+"/param_path_" +str(n_model)+".csv")

	f =open(o_dir,'r')
	temp_output = []
	temp_mean = []
	for l in f:
	    l = l[1:-2]
	    string_l = l.split(',')
	    temp_output.append( [ float(i) for i in string_l] )
	    temp_mean.append( np.mean( [ float(i) for i in string_l] ))
	f.close()

	path = np.genfromtxt(path_dir,delimiter=',') 
	params = np.genfromtxt(p_dir,delimiter=',')

	seen_parameters = [1]
	tot_seen_params = 0
	for k in range(1,length):
	    is_in,idx = is_in_ndarray(path[k,:],params[:(tot_seen_params),:])
	    if(is_in):
	        seen_parameters[idx] += 1
	    else:
	        tot_seen_params += 1
	        seen_parameters.append(1)

	output = []
	m_o = []
	std_o = []
	for l in range(tot_seen_params+1):
	    o = [temp_output[l][x] for x in range(5*seen_parameters[l])]
	    output.append( o )
	    m_o.append(np.mean(o))
	    std_o.append(np.std(o))

	parameters = params[:(tot_seen_params+1),:]
	    
	return parameters,output,m_o,std_o


def is_in_ndarray(item,a):
	k = 0
	idx_val = np.asarray(range(a.shape[0]))
	idxk = range(a.shape[0])
	while( k < a.shape[1]):
	    idxk =  (a[idxk,k]==item[k])
	    if(np.sum(idxk > 0)):
	        k += 1
	        idx_val = idx_val[idxk]
	        idxk = list(idx_val) 
	    else:
	        return False,0

	return True,idx_val[0]



params,output,m_o,std_o = get_exp_data(4001,495)
rand_candidates = utils.sample_random_candidates(1000,parameter_bounds,None,isInt=np.ones(5))

gcp = GaussianCopulaProcess(nugget = 1e-10,
                                corr= 'squared_exponential',
                                random_start=5,
                                n_clusters=1,
                                coef_latent_mapping = 0.1,
                                try_optimize=True)
gcp.fit(params,m_o,output,obs_noise=std_o)

gp = GaussianProcess(theta0= 0.1 ,
                     thetaL = 0.001,
                     random_start=1,
                     thetaU = 10.,
                     nugget=1e-10)
gp.fit(params,m_o)
gp_pred,sigma = gp.predict(rand_candidates,eval_MSE=True)
gp_bu = np.asarray(gp_pred) + 1.96*np.sqrt(sigma)
print gp.theta_

predictions,mse,bl,bu = gcp.predict(rand_candidates,eval_MSE=True,eval_confidence_bounds=True,integratedPrediction= False,coef_bound=.5)

params_bis = (10*params[:,3] +  5* params[:,0] + params[:,1]) / 10.
params_bis2 = params[:,2] + (3.* params[:,4] / 10. )

cand_bis = (10*rand_candidates[:,3] +  5* rand_candidates[:,0] + rand_candidates[:,1]) / 10.
cand_bis2 = rand_candidates[:,2] + (3.* rand_candidates[:,4] / 10. )

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
Z_min = np.min(m_o)
ax.scatter(params_bis2,params_bis,\
                                    m_o, c=-np.exp((m_o-Z_min)**5.),cmap=cm.hot,marker='o')
ax.plot(cand_bis2,cand_bis,predictions,'b+')
ax.plot(cand_bis2,cand_bis,gp_pred,'g+')

plt.show()