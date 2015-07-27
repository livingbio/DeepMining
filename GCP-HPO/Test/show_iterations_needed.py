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

import matplotlib.pyplot as plt
from iterations_needed import iterationsNeeded
import numpy as np

first_exp = 4001
last_exp = 4050
test_name = "MNIST"

threshold = 0.5
alpha = 0.5


mean_iter_needed,q1_iter_needed,median_iter_needed,q3_iter_needed = \
				iterationsNeeded(test_name,first_exp,last_exp,threshold,alpha)

abs = 95 + 0.05 * np.asarray(range(101))

fig = plt.figure(figsize=(15,7))
plt.plot(abs,median_iter_needed,'c')
plt.plot(abs,q1_iter_needed,'c-.')
plt.plot(abs[q3_iter_needed < 1000],q3_iter_needed[q3_iter_needed < 1000],'c-.')
plt.title('Iterations needed')
plt.xlabel('Percentage of maximum gain')
plt.ylabel('Number of tested parameters')
plt.show() 