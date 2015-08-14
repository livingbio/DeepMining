# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

# The MIT License (MIT)
# Copyright (c) 2015 Sebastien Dubois

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import sys

sys.path.append("../../")
sys.path.append("../Branin/")
sys.path.append("../Hartmann/")
from branin import branin
from har6 import har6
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import NearestNeighbors

### Set parameters ###
nugget = 1.e-10
n_clusters_max = 3
integratedPrediction = False
n_tests = 20
coef_latent_mapping = 0.1
save_data = True
isInt = True
print 'Average on n_tests = 20, nugget = 1.e-10, integratedPrediction = False,coef_latent_mapping = 0.1'

def artificial_f(x):
	x = x[0]
	res = (70-7*np.exp(x/50. - ((x-55.)**2)/500.) + 6*np.sin(x/40.) +3./(1.1+np.cos(x/50.)) - 15./(3.3-3*np.sin((x-70)/25.)))/100.
	return [res]

def branin_f(p_vector):
	x,y = p_vector
	x = x -5.
	y= y
	return branin(x,y)

# MNIST data
mnist_output = []
f =open(("../MNIST/scoring_function/output.csv"),'r')
for l in f:
  l = l[1:-3]
  string_l = l.split(',')
  mnist_output.append( [ float(i) for i in string_l] )
f.close()
mnist_params = np.genfromtxt(("../MNIST/scoring_function/params.csv"),delimiter=',')
mnist_KNN = NearestNeighbors()
mnist_KNN.fit(mnist_params)

# Popcorn data
popcorn_output = []
f =open(("../Bags_of_Popcorn/scoring_function/output.csv"),'r')
for l in f:
  l = l[1:-3]
  string_l = l.split(',')
  popcorn_output.append( [ float(i) for i in string_l] )
f.close()
popcorn_params = np.genfromtxt(("../Bags_of_Popcorn/scoring_function/params.csv"),delimiter=',')
popcorn_KNN = NearestNeighbors()
popcorn_KNN.fit(popcorn_params)

# function that retrieves a performance evaluation from the stored results
def mnist_f(p):
	idx = mnist_KNN.kneighbors(p,1,return_distance=False)[0]
	all_o = mnist_output[idx]
	# r = np.random.randint(len(all_o)/5)
	# return all_o[(5*r):(5*r+5)]
	return [ np.mean(all_o)]

def popcorn_f(p):
	idx = popcorn_KNN.kneighbors(p,1,return_distance=False)[0]
	all_o = popcorn_output[idx]
	return [ np.mean(all_o)]

all_parameter_bounds = {'artificial_f': np.asarray( [[0,400]] ),
						'branin': np.asarray( [[0,15],[0,15]] ),
						'har6' : np.asarray( [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]),
						 'mnist' : np.asarray( [[0,2],[0,5],[5,31],[1,5],[0,4]] ),
						 'popcorn' : np.asarray( [[13,33],[5,11],[1,5],[4,11],[0,2],[1,11],[1,4]] ) }

training_sizes = {'artificial_f': [30,40,50,60,70,80],
						'branin': [5,8,11,14,17,20] ,
						'har6' : [50,80,100,150] , 
						'mnist' : [200,300,400],
						'popcorn' : [400,500,600]}
						# 'har6' : [30,35,40,45,50,55,60]  }

prediction_sizes = {'artificial_f': [1000],
						'branin': [1000] ,
						'har6' : [1000] ,
						'mnist' : [1000],
						'popcorn' : [1000] }

tests = {'function' : [ 'popcorn' ],
		'corr_kernel': ['exponential_periodic'] }
# 		'corr_kernel': ['squared_exponential','exponential_periodic'] }

functions = {'artificial_f':artificial_f,'branin':branin_f,'har6':har6, 'mnist':mnist_f, 'popcorn':popcorn_f}

files ={}
if(save_data):
	for function in tests['function']:
		# file = open(function + '_pred1000_20tests.csv','w')
		# file.write('Training size,Likelihood_GP,lkh_err_GP,SMSE_GP,smse_err_GP,\
		# 	Likelihood_GCP_SE,lkh_err_GCP_SE,SMSE_GCP_SE,smse_err_GCP_SE,\
		# 	Likelihood_GCP_EP,lkh_err_GCP_EP,SMSE_GCP_EP,smse_err_GCP_EP\n')
		file = open(function +'_LGCP_EP_coef1_pred1000_20tests_3.csv','w')
		file.write('Training size,Likelihood_GP,lkh_err_GP,SMSE_GP,smse_err_GP')
		for n in range(n_clusters_max):
			file.write(',Likelihood_' + str(n) \
				+ ',lkh_err_' + str(n) \
				+ ',SMSE_' + str(n) \
				+ ',smse_err_' + str(n))
		file.write('\n')
		files[function] = file

def compute_unique2(a1,a2):
	# keep only unique rows of a1, and delete the corresponding rows in a2
	
	b = np.ascontiguousarray(a1).view(np.dtype((np.void, a1.dtype.itemsize * a1.shape[1])))
	_, idx = np.unique(b, return_index=True)
	idx =np.sort(idx)
	return a1[idx],a2[idx]	

def run_test(training_size,prediction_size,function_name,corr_kernel,n_cluster,prior='GCP'):

	scoring_function = functions[function_name]
	parameter_bounds = all_parameter_bounds[function_name]

	x_training = []
	y_training = []
	for i in range(training_size):
		x = [np.random.uniform(parameter_bounds[j][0],parameter_bounds[j][1]) for j in range(parameter_bounds.shape[0])]
		x_training.append(x)
		y_training.append(scoring_function(x)[0])
	if(isInt):
		x_training,y_training = compute_unique2( np.asarray(x_training,dtype=np.int32) , np.asarray( y_training) )

	candidates = []
	real_y = []
	for i in range(prediction_size):
		x = [np.random.uniform(parameter_bounds[j][0],parameter_bounds[j][1]) for j in range(parameter_bounds.shape[0])]
		candidates.append(x)
		real_y.append(scoring_function(x)[0])
	real_y = np.asarray(real_y)
	if(isInt):
		candidates = np.asarray(candidates,dtype=np.int32)

	if(prior == 'GP'):
		gp = GaussianProcess(theta0=.1 *np.ones(parameter_bounds.shape[0]),
						 thetaL = 0.001 * np.ones(parameter_bounds.shape[0]),
						 thetaU = 10. * np.ones(parameter_bounds.shape[0]),
						 random_start = 5,
						 nugget=nugget)
		gp.fit(x_training,y_training)
		pred = gp.predict(candidates)
		likelihood = gp.reduced_likelihood_function_value_

	else:
		gcp = GaussianCopulaProcess(nugget = nugget,
		                            corr=corr_kernel,
		                            random_start=5,
		                            normalize = True,
		                            coef_latent_mapping = coef_latent_mapping,
		                            n_clusters=n_clusters)
		gcp.fit(x_training,y_training)
		likelihood = gcp.reduced_likelihood_function_value_
		
		if not (integratedPrediction):
			pred = gcp.predict(candidates)
		else:
			pred,_,_,_ = gcp.predict(candidates,eval_MSE = True,eval_confidence_bounds=True,integratedPrediction=True)

	mse = np.mean( (pred - real_y)**2. )
	# Normalize 
	mse = mse / ( np.std(real_y) **2. )
	likelihood = np.exp(likelihood)
	return [mse,likelihood]


for f in tests['function']:
	print ' **  Test function',f,' ** '
	if(save_data):
		file = files[f]
	for t_size in training_sizes[f]:
		for p_size in prediction_sizes[f]:
			print 'Training size :',t_size,'- Prediction size :',p_size
			res = np.asarray( [run_test(t_size,p_size,f,'',0,prior='GP') for j in range(n_tests)] )
			print '\t\t\t\t GP - Lklhood =',np.mean(res[:,1]),'\t',np.std(res[:,1]),'\t- MSE =',np.mean(res[:,0]),'\t',np.std(res[:,0]),'\n'
			if(save_data):
				# file.write(str(t_size) + ',')
				file.write(str(t_size) + ',' + str(np.mean(res[:,1])) + ',' + str(np.std(res[:,1])) + ',' + str(np.mean(res[:,0])) + ',' + str(np.std(res[:,0]) ) +',' )
			for k in tests['corr_kernel']:
				for n_clusters in range(1,n_clusters_max):
					res = np.asarray( [run_test(t_size,p_size,f,k,n_clusters) for j in range(n_tests)] )
					idx = (res[:,0] < 10000)
					if( np.sum(idx) != n_tests):
						print('Warning : only ' + str(np.sum(idx)) +' tests passed')
						res = res[idx,:]
					print 'corr_kernel:',k,'- n_clusters:',n_clusters,'\t- Lklhood =',np.mean(res[:,1]),'\t',np.std(res[:,1]),'\t- MSE =',np.mean(res[:,0]),'\t',np.std(res[:,0])
					if(save_data):
						file.write( str(np.mean(res[:,1])) + ',' + str(np.std(res[:,1])) + ',' + str(np.mean(res[:,0])) + ',' + str(np.std(res[:,0]) ) +',' )
				if(save_data):
					file.write('\n')
		print ''
	print ''
print ''


