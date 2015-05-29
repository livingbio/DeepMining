# -*- coding: utf-8 -*-

# Author: Sebastien Dubois 
#		  for ALFA Group, CSAIL, MIT

import numpy as np
from sklearn_utils import *

def GCP_Xwrapping(X,x_wrapping):
	## do domething
	return X


def theta_toOneDim(theta):
	if(theta.shape[1] == 1):
		return theta[:,0]
	if(theta.shape[0] > 1):
		# we don't want t0,t1,t2 to varry with the axis
		theta_1dim = np.mean( theta[:3,:],axis=1) #take the mean in case there has been a error somewhere
		theta_1dim = np.concatenate((theta_1dim, (theta[3:,:]).reshape(1,theta.size-3*theta.shape[1])[0] ))
		return theta_1dim
	else:
		return theta[0,:]
	
	
def theta_backToRealShape(theta_1dim,theta_shape):
	temp = []
	for i in range(theta_shape[1]):
		temp.append(theta_1dim[:3])
	theta = ( np.asarray(temp)).T	
	theta = np.concatenate((theta, (theta_1dim[3:]).reshape([theta_shape[0]-3,theta_shape[1]]) ))
	
	return theta

	
def find_bounds(f, y):
	x = 1
	while((f(x) < y)  and (x<2047483646)):
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
	delta = np.float(hi-lo)/10000.
	while lo <= hi:
		x = (lo + hi) / 2
		#print(x)
		if f(x) < y:
			lo = x + delta
		elif f(x) > y:
			hi = x - delta
		else:
			return x 
	if (f(hi) - y < y - f(lo)):
		return hi
	else:
		return lo	
		

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
	return np.exp( - np.sum( theta.reshape(1, theta.size) * d ** 2, axis=1)  )


def exponential_periodic(theta,d):
	# theta is a numpy array with shape (8,1) or (8,d.shape[1])
	t0 = np.mean(theta[0,:] ) / 100.
	t1 = np.mean(theta[1,:] ) / 100.
	t2 = np.mean(theta[2,:] ) / 100.
	t3 = theta[3,:]
	t4 = theta[4,:]
	t5 = theta[5,:]
	t6 = theta[6,:]
	t7 = theta[7,:]
	t8 = theta[8,:]
	#print(theta)
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
		#print ( np.min(((c1+c2+c3)/c)))
		#print (np.max(((c1+c2+c3)/c)))
		return ((c1+c2+c3)/c)
	
	else:
		print('Corr Error 2')
		return np.asarray([0.])
