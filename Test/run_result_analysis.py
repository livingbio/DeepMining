import matplotlib.pyplot as plt
import sys
from analyze_results import analyzeResults

test_name = "MNIST"
# threshold to make the means setps through Welch's t-test
threshold = 0.5
# coef to compute the final score : mean - alpha * std
alpha = 0.5

n_exp = 5001
verbose = True

scores = analyzeResults(test_name,n_exp,threshold,alpha,verbose)

fig = plt.figure(figsize=(15,7))
plt.plot(range(scores.shape[0]),scores)
plt.show() 