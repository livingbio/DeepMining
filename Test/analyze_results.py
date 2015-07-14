import os
import numpy as np
from scipy import stats


def analyzeResults(test_name,n_exp,threshold,alpha,verbose):

    folder = test_name + "/exp_results/transformed_t_"+str(threshold)+"_a_"+str(alpha)
    if not os.path.exists(folder):
      os.mkdir(folder)

    if os.path.exists(folder+"/exp"+str(n_exp)+".csv"):
        score = np.genfromtxt(folder+"/exp"+str(n_exp)+".csv",delimiter=',')
        return score


    mean_outputs = []
    std_outputs = []
    raw_outputs = []

    p_dir = test_name + "exp_results/exp"+str(n_exp)+"/param_0.csv"
    f =open(test_name + "exp_results/exp"+str(n_exp)+"/output_0.csv",'r')

    for l in f:
        l = l[1:-2]
        string_l = l.split(',')
        raw_outputs.append( [ float(i) for i in string_l] )
    f.close()

    c_print = 0
    for o in raw_outputs:
        mean_outputs.append(np.mean(o))
        if(verbose):
            print c_print,np.mean(o)
            c_print += 1
        std_outputs.append(np.std(o))


    parameters = np.genfromtxt(p_dir,delimiter=',')

    sorted_idx = np.argsort(mean_outputs)
    sorted_raw_outputs= [ raw_outputs[i] for i in sorted_idx ]

    # Cluster with Welch's t-test
    clustered_obs = []
    clusters_param_idx = []
    c_count = -1
    for i in range( len( sorted_raw_outputs) ):
        o = sorted_raw_outputs[i]
        if(c_count > -1 ):
            t,p = stats.ttest_ind(clustered_obs[c_count],o,equal_var=False)
            if(p < threshold):
                clustered_obs.append(np.copy(o))
                clusters_param_idx.append( [sorted_idx[i]] )
                c_count += 1
            else:
                clustered_obs[c_count] = np.concatenate((clustered_obs[c_count], np.copy(o)))
                clusters_param_idx[c_count] += [sorted_idx[i]]
        else:
            clustered_obs.append(np.copy(o))
            clusters_param_idx.append([ sorted_idx[i] ] )
            c_count = 0
        
    if(verbose):        
        print '\n', len(clustered_obs),' clusters'
        for i in range(len(clustered_obs)):
            c_o = clustered_obs[i]
            print 'Size',len(c_o),'\t','Mean',np.mean(c_o)
            print 'Params idx', clusters_param_idx[i]
        print '\n'


    # compute score
    params_idx = []
    params_results = []
    for i in range(1,len(clustered_obs)+1):
        mean_on_cluster = np.mean(clustered_obs[-i])
        for idx in clusters_param_idx[-i]:
            params_results.append( [ mean_on_cluster ,np.std(raw_outputs[idx]) ] )
        params_idx += clusters_param_idx[-i]
    params_results = np.asarray(params_results)

    final_score = params_results[:,0] - alpha*params_results[:,1]
                
    final_sorted_idx = np.argsort(final_score)[::-1]
    #print final_score
    rank_idx = np.asarray(params_idx)[final_sorted_idx]
    if(verbose):
        print 'Sorted params idx:\n',rank_idx 
        best_idx = params_idx[final_sorted_idx[0]]
        print '\nBest parameter set', parameters[best_idx,:]
        print 'Next:'
        for i in range(1,5):
            print parameters[params_idx[final_sorted_idx[i]]]

    parameter_score = []
    idx_matching = np.argsort(params_idx)
    #path_params = np.genfromtxt(path_dir,delimiter=',')
    for i in range(parameters.shape[0]):
        parameter_score.append(final_score[idx_matching[i]])

    np.savetxt(folder+"/exp"+str(n_exp)+".csv",parameter_score,delimiter=',')

    return np.asarray(parameter_score)
