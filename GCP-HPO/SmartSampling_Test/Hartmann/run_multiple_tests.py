import numpy as np
import sys
import os
sys.path.append("../../")
from smart_sampling import smartSampling
from har6 import har6

first_exp = 301
n_exp = 10

parameter_bounds = np.asarray( [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]] )
model = 'GCP'
n_random_init = 20
n_smart_iter = 175
n_candidates=500
corr_kernel='squared_exponential'
acquisition_function = 'MaxUpperBound'
n_clusters = 1
cluster_evol = 'variable'
GCP_mapWithNoise=False
GCP_useAllNoisyY=False
model_noise=None
nugget = 1.e-10

last_exp = first_exp + n_exp
print 'Run experiment',first_exp,'to',last_exp



###  Run experiment  ### 

for n_exp in range(first_exp,last_exp):
    print ' ****   Run exp',n_exp,'  ****'
    ### set directory
    if not os.path.exists("exp_results/exp"+str(n_exp)):
        os.mkdir("exp_results/exp"+str(n_exp))
    else:
        print('Warning : directory already exists')

    all_parameters,all_raw_outputs,all_mean_outputs, all_std_outputs, all_param_path = \
        smartSampling(n_smart_iter,parameter_bounds,har6,isInt=False,
                      corr_kernel = corr_kernel ,
                      GCP_mapWithNoise=GCP_mapWithNoise,
                      GCP_useAllNoisyY=GCP_useAllNoisyY,
                      model_noise = model_noise,
                      model = model, 
                      n_candidates=n_candidates,
                      n_random_init=n_random_init, 
                      n_clusters=n_clusters,
                      cluster_evol = cluster_evol,
                      verbose=True,
                      nugget = nugget,
                      acquisition_function = acquisition_function,
                      detailed_res = True)

    ## save experiment's data
    for i in range(len(all_raw_outputs)):
        f =open(("exp_results/exp"+str(n_exp)+"/output_"+str(i)+".csv"),'w')
        for line in all_raw_outputs[i]:
            print>>f,line
        f.close()
        np.savetxt(("exp_results/exp"+str(n_exp)+"/param_"+str(i)+".csv"),all_parameters[i], delimiter=",")
        np.savetxt(("exp_results/exp"+str(n_exp)+"/param_path_"+str(i)+".csv"),all_param_path[i], delimiter=",")

    print ' ****   End experiment',n_exp,'  ****\n'
