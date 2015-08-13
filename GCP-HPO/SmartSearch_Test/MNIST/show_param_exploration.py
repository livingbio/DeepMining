import numpy as np
import sys

sys.path.append("../../")
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

name = ['GCP_UCB_SE','LGCP_UCB_EP','GP','random']
first_exp = [13001,4001,20501,1]
last_exp = [13050,4050,20560,200]
all_counts = []
all_coords = []
all_scores = []

for j in range(len(first_exp)):
    all_x = np.array([])
    all_y = np.array([])
    for n_exp in range(first_exp[j],last_exp[j]):
    	if(name[j] == 'random'):
	        params,output,m_o,std_o = get_exp_data(n_exp,450,isRand=True)
    	else:
	        params,output,m_o,std_o = get_exp_data(n_exp,450,isRand=False)    		
        params_bis = (10*params[:,3] +  5* params[:,0] + params[:,1]) / 10.
        params_bis2 = params[:,2] + (3.* params[:,4] / 10. )
        all_x = np.concatenate((all_x,params_bis))
        all_y = np.concatenate((all_y,params_bis2))
    collec = plt.hexbin(all_x,all_y,gridsize=20,vmin=0,vmax=500,cmap=plt.cm.afmhot_r,mincnt=1)
    counts = collec.get_array()
    coord = collec.get_offsets()
    print coord.shape
    scores = []
    for i in range(coord.shape[0]):
        neighbor_idx = mnist_KNN.kneighbors(coord[i,:],1,return_distance=False)[0]
        scores.append(mnist_output[neighbor_idx])
    print max(counts)
    print len(scores)
    nb_trials = last_exp[j] -  first_exp[j]
    counts = counts / nb_trials

    all_counts.append(counts)
    all_coords.append(coord)
    all_scores.append(list(scores))

    f=open('data_plots/param_exploration/' + name[j] +'.csv','w')
    f.write('x,y,score,freq\n')
    for k in range(counts.shape[0]):
    	f.write( str(coord[k,0]) + ',' + str(coord[k,1]) + ',' + str(scores[k]) + ',' + str(counts[k])  +'\n')