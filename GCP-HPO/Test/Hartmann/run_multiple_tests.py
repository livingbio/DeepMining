import numpy as np
import sys
import os
sys.path.append("../../")
from har6 import har6
from smart_search import SmartSearch

first_exp = 301
n_exp = 10

parameters = { 'a' : ['float',[0,1]],
         'b' : ['float',[0,1]],
         'c' : ['float',[0,1]],
         'd' : ['float',[0,1]],
         'e' : ['float',[0,1]],
         'f' : ['float',[0,1]] }
model = 'GCP'
n_random_init = 20
n_iter = 200
n_candidates=500
corr_kernel='squared_exponential'
acquisition_function = 'UCB'
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
    search = SmartSearch(parameters,
          estimator=scoring_function,
          corr_kernel = corr_kernel,
          acquisition_function = acquisition_function,
          GCP_mapWithNoise=mapWithNoise,
          model_noise = model_noise,
          model = sampling_model, 
          n_candidates=n_candidates,
          n_iter = n_iter,
          n_init = n_random_init,
          n_final_iter=nb_iter_final,
          n_clusters=n_clusters, 
          cluster_evol = cluster_evol,
          verbose=2,
          detailed_res = 2)
    all_parameters, all_search_path, all_raw_outputs,all_mean_outputs = search._fit()

    ## save experiment's data
    for i in range(len(all_raw_outputs)):
        f =open(("exp_results/exp"+str(n_exp)+"/output_"+str(i)+".csv"),'w')
        for line in all_raw_outputs[i]:
            print>>f,line
        f.close()
        np.savetxt(("exp_results/exp"+str(n_exp)+"/param_"+str(i)+".csv"),all_parameters[i], delimiter=",")
        np.savetxt(("exp_results/exp"+str(n_exp)+"/param_path_"+str(i)+".csv"),all_search_path[i], delimiter=",")

    print ' ****   End experiment',n_exp,'  ****\n'
