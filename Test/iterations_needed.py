import os
import numpy as np
from sklearn.neighbors import NearestNeighbors


def iterationsNeeded(test_name,first_exp,last_exp,threshold,alpha):

	result_path = test_name + "/exp_results/iterations_needed/exp" + str(first_exp) + "_" + str(last_exp) + "_t_" +str(threshold)+"_a_"+str(alpha) +".csv"
	folder = test_name + "/exp_results/iterations_needed"
	if not os.path.exists( folder):
		os.mkdir( folder )

	if os.path.exists(result_path):
		result = np.genfromtxt(result_path,delimiter=',')
		return result

	p_dir = test_name + "/scoring_function/params.csv"
	scores = np.genfromtxt(test_name + "/scoring_function/true_score_t_"+str(threshold)+"_a_"+str(alpha) + ".csv",delimiter=',')
	all_params = np.genfromtxt(p_dir,delimiter=',')
	KNN = NearestNeighbors()
	KNN.fit(all_params)

	m = np.min(scores)
	M = np.max(scores)

	all_iter_needed = []

	for n_exp in range(first_exp,last_exp+1):
		
		path = np.genfromtxt(test_name + "/exp_results/exp"+str(n_exp)+"/param_path_0.csv",delimiter=',')
		true_score = np.zeros(path.shape[0])
		for i in range(path.shape[0]):
			neighbor_idx = KNN.kneighbors(path[i,:],1,return_distance=False)[0]
			true_score[i] = 100.* (scores[neighbor_idx]-m)/(M-m)

		n_iter_needed =  np.zeros(101)

		starting_score = 95.
		nb_iter = 0
		for i in range(101):
			while(nb_iter < path.shape[0] and true_score[nb_iter] < starting_score + 0.05*i):
				nb_iter += 1
			if(nb_iter == path.shape[0]):
				print 'Exp is too short'
				nb_iter = 1500
			n_iter_needed[i] = nb_iter

		all_iter_needed.append(n_iter_needed)

	mean_iter_needed = [np.mean([all_iter_needed[j][i] for j in range(last_exp+1-first_exp)]) for i in range(101) ]
	median_iter_needed = [np.median([all_iter_needed[j][i] for j in range(last_exp+1-first_exp)]) for i in range(101) ]
	q1_iter_needed = [np.percentile([all_iter_needed[j][i] for j in range(last_exp+1-first_exp)],q=25) for i in range(101) ]
	q3_iter_needed = [np.percentile([all_iter_needed[j][i] for j in range(last_exp+1-first_exp)],q=75) for i in range(101) ]

	all_results = np.concatenate((np.atleast_2d(mean_iter_needed),np.atleast_2d(q1_iter_needed), \
											np.atleast_2d(median_iter_needed),np.atleast_2d(q3_iter_needed)))

	np.savetxt(result_path,all_results,delimiter=',')

	return all_results

