import numpy as np
import sys

sys.path.append("../../")
from gcp import GaussianCopulaProcess
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

# MNIST data
mnist_output = []
outputs = []
f =open(("scoring_function/output.csv"),'r')
for l in f:
  l = l[1:-3]
  string_l = l.split(',')
  mnist_output.append( [ float(i) for i in string_l] )
  outputs.append(np.mean([ float(i) for i in string_l] ))
f.close()
outputs = np.asarray(outputs)
mnist_params = np.genfromtxt(("scoring_function/params.csv"),delimiter=',')

params_bis = (10*mnist_params[:,3] +  5* mnist_params[:,0] + mnist_params[:,1]) / 10.
params_bis2 = mnist_params[:,2] + (3.* mnist_params[:,4] / 10. )

Z_min = np.min(outputs)

save_file = open('data_plots/scoring_function.csv','w')
save_file.write('x,y,z,meta\n')
for i in range(outputs.shape[0]/20):
	j = i*20
	save_file.write(str(params_bis[j]) + ',' + str(params_bis2[j]) \
		+ ',' + str(outputs[j]) + ',' + str(-np.exp((outputs[j]-Z_min)**11.)) + '\n' )

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
surf = ax.scatter(params_bis,params_bis2, outputs, c=-np.exp((outputs-Z_min)**11.),cmap=cm.hot,marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()