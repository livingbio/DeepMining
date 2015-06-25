#36001
import numpy as np
import sys
nb_exp = 5
first_exp = int(sys.argv[1])

print 'Arguments:',sys.argv
last_exp = first_exp + nb_exp
print 'Run exp',first_exp,'to',last_exp


GCPconsiderAllObs1= True
GCPconsiderAllObs2= False
noise_restitution = 'rgni'
model = 'GCPR'
nb_parameter_sampling= 500
nb_random_steps= 20
cluster_evol = 'constant'
acquisition_function = 'MaxUpperBound'
corr_kernel = #'exponential_periodic' #'squared_exponential' 

pop_size = 5000
exp_dir = "all_pop5000"

o_fnames = [(exp_dir+"_output")]

p_fnames = [(exp_dir+"_param")]

# nb_features, feat_select,max_ngram_range,max_df, min_df, alpha_NB, mode_tfidf
parameter_bounds = np.asarray( [
        [13,33],
        [5,11],
        [1,5],
        [4,11],
        [0,2],
        [1,11],
        [1,4]] )

nb_GCP_steps = 90


from scipy.spatial.distance import euclidean
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import stats
import os
from sklearn.neighbors import NearestNeighbors


import sys
sys.path.append("../../")
from smart_sampling import smartSampling



####### Prepare FastScoring

output = []
for fname in o_fnames:
    print fname
    f =open(("scoring_function/"+fname+".csv"),'r')
    for l in f:
        l = l[1:-3]
        string_l = l.split(',')
        output.append( [ float(i) for i in string_l] )
    f.close()
    print len(output)


params = np.genfromtxt(("scoring_function/"+p_fnames[0]+".csv"),delimiter=',')
print params.shape


KNN = NearestNeighbors()
KNN.fit(params)
# KNN.kneighbors(p,1,return_distance=False)[0]



def get_cv_res(p):
    idx = KNN.kneighbors(p,1,return_distance=False)[0]
    all_o = output[idx]
    r = np.random.randint(len(all_o)/5)
    return all_o[(5*r):(5*r+5)]


####### Run exp ####### 

for n_exp in range(first_exp,last_exp):
    print ' ****   Run exp',n_exp,'  ****'
    ### set directory
    if not os.path.exists("fast_exp_results/"+exp_dir+"/fexp"+str(n_exp)):
        os.mkdir("fast_exp_results/"+exp_dir+"/fexp"+str(n_exp))
    else:
        print('Be carefull, directory already exists')

    all_parameters,all_raw_outputs,all_mean_outputs, all_std_outputs, all_param_path = \
        smartSampling(nb_GCP_steps,parameter_bounds,get_cv_res,isInt=True,
                      corr_kernel = corr_kernel ,
                      GCPconsiderAllObs1=GCPconsiderAllObs1,
                      GCPconsiderAllObs2=GCPconsiderAllObs2,
                      noise_restitution=noise_restitution,
                      model = 'GCPR', nb_parameter_sampling=nb_parameter_sampling,
                      nb_random_steps=nb_random_steps, n_clusters=1,cluster_evol = cluster_evol,
                      verbose=True,
                      acquisition_function = acquisition_function)

    ## save exp
    for i in range(len(all_raw_outputs)):
        f =open(("fast_exp_results/"+exp_dir+"/fexp"+str(n_exp)+"/output_"+str(i)+".csv"),'w')
        for line in all_raw_outputs[i]:
            print>>f,line
        f.close()
        np.savetxt(("fast_exp_results/"+exp_dir+"/fexp"+str(n_exp)+"/param_"+str(i)+".csv"),all_parameters[i], delimiter=",")
        np.savetxt(("fast_exp_results/"+exp_dir+"/fexp"+str(n_exp)+"/param_path_"+str(i)+".csv"),all_param_path[i], delimiter=",")

    print ' ****   End exp',n_exp,'  ****\n'
