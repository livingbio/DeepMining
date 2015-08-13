import numpy as np
import matplotlib.pyplot as plt

labels = ['GCP','GP','GCP-EP','LGCP']
first_exps = [1,101,201,301]
last_exps = [10,110,207,307]

fig = plt.figure()

for i in range(len(first_exps)):
	all_max_outputs = []
	for n_exp in range(first_exps[i],last_exps[i]+1):
		
		f =open("exp_results/exp"+str(n_exp)+"/output_0.csv",'r')
		output = []
		for l in f:
		    l = l[1:-2]
		    output.append( float(l) )
		f.close()

		max_output = np.zeros(len(output))
		max_output[0] = output[0]
		for j in range(1,len(output)):
			max_output[j] = max(max_output[j-1],output[j])

		all_max_outputs.append(max_output)

	mean_result = [ np.mean([all_max_outputs[k][j] for k in range(len(all_max_outputs))] ) for j in range(len(all_max_outputs[0]))]
	std_result = [ np.std([all_max_outputs[k][j] for k in range(len(all_max_outputs))] ) for j in range(len(all_max_outputs[0]))]
	print labels[i],'max:', np.max([all_max_outputs[k][-1] for k in range(len(all_max_outputs))] ) ,\
					'mean:',mean_result[-1],'std:',std_result[-1]
	
	plt.plot(range(len(mean_result)),mean_result,label=labels[i])

plt.legend()
plt.show()