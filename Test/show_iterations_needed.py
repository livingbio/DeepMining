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