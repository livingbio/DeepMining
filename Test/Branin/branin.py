import numpy as np
import math

"""
The Branin function is a classical example to evaluate nonlinear optimization algorithms.
The goal is to find the minimum, we use here the opposite of the Branin function to match
the maximization process.
"""

# x in [-5,10] -  y in [0,15]
# min is 0.398

def branin(x, y):
	"""
	The opposite of the Branin function
	"""

	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10

	return [-result]


def noisy_branin(x, y):
	"""
	The opposite of the Branin function with an additive Gaussian noise
	"""
	
	result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + \
		(5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10

	result += np.random.normal() * 50.

	return [-result]