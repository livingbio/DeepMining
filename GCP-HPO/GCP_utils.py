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
from sklearn_utils import *


def theta_toOneDim(theta):
	"""
	When using the Exponential Periodic kernel, theta should be correctly reshaped
	so that the first three coefficients are the same for all dimensions.
	This method reshapes the (n_kernel_parameters,n_features) theta into a one dimensional
	vector.
	"""
	if(theta.shape[1] == 1):
		return theta[:,0]
	if(theta.shape[0] > 1):

		# we don't want t0,t1,t2 to varry with the axis
		theta_1dim = np.mean( theta[:3,:],axis=1) #take the mean in case there has been an error somewhere
		theta_1dim = np.concatenate((theta_1dim, (theta[3:,:]).reshape(1,theta.size-3*theta.shape[1])[0] ))
		return theta_1dim
	else:
		return theta[0,:]
	
	
def theta_backToRealShape(theta_1dim,theta_shape):
	"""
	When using the Exponential Periodic kernel, theta should be correctly reshaped
	so that the first three coefficients are the same for all dimensions.
	This method restores the real shape (n_kernel_parameters,n_features).
	"""
	if(theta_shape[0] > 1):
		temp = []
		for i in range(theta_shape[1]):
			temp.append(theta_1dim[:3])
		theta = ( np.asarray(temp)).T	
		theta = np.concatenate((theta, (theta_1dim[3:]).reshape([theta_shape[0]-3,theta_shape[1]]) ))
		
		return theta
	else:
		return theta_1dim

	
def find_bounds(f, y):
	# to invert a function by binomial search

	x = 1
	while((f(x) < y)  and (x<1000483646) ):
		x = x * 2
	lo = -100 
	if (x ==1):
		lo = -100
	else:
		lo = x/2
	if(x > 1000):
		x = min(x,2047483646)
	return lo, x
	
def binary_search(f, y, lo, hi):
	# to invert a function by binomial search

	delta = np.float(hi-lo)/1000000.
	while lo <= hi:
		x = (lo + hi) / 2
		#print(x)
		if f(x) < y:
			lo = x + delta
		elif f(x) > y:
			hi = x - delta
		else:
			return x 
	if (f(hi)[0] - y < y - f(lo)[0]):
		return hi
	else:
		return lo	
		
def listOfList_toArray(params,obs):
	# converts a list of lists (obs) into a one dimensional array 
	# while keeping all the values in obs and repeating the parameters
	# so that the index correspondence between params and obs is kept

	array_obs = []
	all_params = []
	for i in range(len(obs)):
		p = params[i]
		for o in obs[i]:
			array_obs.append(o)
			all_params.append(p)
	array_obs = np.asarray(array_obs)
	all_params = np.asarray(all_params)

	return all_params,array_obs

def reshape_cluster_labels(labels,detailed_X):
	# reshape the list labels that matched to the params array
	# (containing unique values) to all_params array (parameters 
	# are repated to match detailed_obs)
	# Note that repeated params appear successively

	detailed_labels = [labels[0]]
	unique_count = 0 # to map to 'labels' list
	for i in range(1,detailed_X.shape[0]):
		if( any(detailed_X[i] != detailed_X[i-1])):
			unique_count += 1
		detailed_labels.append(labels[unique_count])

	return np.asarray(detailed_labels)


def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X: array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = array2d(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij
	
	
def sq_exponential(theta,d):
	return np.exp( - np.sum( theta * d ** 2, axis=1)  )


def exponential_periodic(theta,d):
	# theta is a numpy array with shape (9,1) or (9,d.shape[1])
	t0 = np.mean(theta[0,:] ) / 100.
	t1 = np.mean(theta[1,:] ) / 100.
	t2 = np.mean(theta[2,:] ) / 100.
	t3 = theta[3,:]
	t4 = theta[4,:]
	t5 = theta[5,:]
	t6 = theta[6,:]
	t7 = theta[7,:]
	t8 = theta[8,:]

	good_cond =  (t0 > 0) and (t1 > 0) and (t2 > 0)
	c = (t0 + t1 + t2) * 5.
	if(good_cond):
		temp1 = 0.
		temp21 = 0.
		temp22 = 0.
		c3 = t2
		for k in range(d.shape[1]):
			temp1 += t3[k] * d[:,k] ** 2
			temp21 += (d[:,k]**2)/(2.* t4[k]**2) 
			temp22 += (np.sin(3.14 *  d[:,k] /t8[k]) /t5[k])**2
			c3 *= ( (1+ (d[:,k]/t7[k])**2 ) )** (-t6[k]) 
		c1 = t0 * np.exp( - temp1)
		c2 = t1 * np.exp( -temp21 - (2.*temp22) )
		
		if( np.sum( ((c1+c2+c3)/c) >= 1. ) >= 1):
			print('Corr Error 1')
			return np.zeros((d.shape[0]))
			
		return ((c1+c2+c3)/c)
	
	else:
		print('Corr Error 2')
		return np.asarray([0.])
