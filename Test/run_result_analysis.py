import matplotlib.pyplot as plt
import sys
from analyze_results import analyzeResults

n_exp = 1
test_name = "MNIST"

threshold = 0.5
alpha = 0.5

smoothQ = True
kneighbors_s = 3
sigma_s = 200.
beta = 5.

verbose = True

scores = analyzeResults(test_name,n_exp,threshold,alpha,
						smoothQ = smoothQ,
						kneighbors_s = kneighbors_s,
						sigma_s = sigma_s,
						beta = beta ,
						verbose=verbose)

fig = plt.figure(figsize=(15,7))
plt.plot(range(scores.shape[0]),scores)
plt.show() 