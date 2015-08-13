## Experiments index ##
If not specified otherwise, a GCP prior is used (n_clusters == 1), GCP_mapWithNoise == False, GCP_useAllNoisyY == False, model_noise == None.
For LGCP priors, +1/N means that the number of clusters is incremented every N *smart* iterations; and n_cluster_max was set to 5.
All experiments were run with 10 iterations for the random initialization, 485 smart iterations, and 5 final ones; except for GP experiments that have 785 smart iterations.

  -----------

Notations :
- Acquisition function : EI (Expected Improvement), UCB (Upper Confidence Bound)

- Correlation kernel : EP (Exponential periodic), SE (Squared exponential)

- Model noise : EGN (Estimated Gaussian Noise)

-----------

- 1000 .............................................................................EI, EP
- 1500 ....................................................EI, EP, mapWithNoise
- 2000 ........................................................LGCP+1/30, EI, EP 
- 3000 ..........................................................................UCB, EP
- 4000 .....................................................LGCP+1/30, UCB, EP
- 4100 .....................................................LGCP+1/10, UCB, EP
- 4200 .....................................................LGCP+1/20, UCB, EP
- 4500 ..........................LGCP+1/30, UCB, EP, , mapWithNoise
- 6000 ..................................................UCB, EP, mapWithNoise
- 11000 ............................................................................EI, SE
- 11500 ....................................................EI, SE, mapWithNoise
- 12000 .......................................................LGCP+1/30, EI, SE
- 13000 ........................................................................UCB, SE
- 14000 ...................................................LGCP+1/30, UCB, SE
- 14500 ...........................LGCP+1/30, UCB, SE, mapWithNoise
- 15000 ..................................................UCB, SE, useAllNoisyY 
- 16000 ...............................................UCB, SE, mapWithNoise
- 21000 ..................................................................GP, UCB, SE
- 31000 .........................................EI, EP, model_noise == EGN
- 31500 .........................................EI, SE, model_noise == EGN
- 32000 .....................LGCP+1/30, EI, EP, model_noise == EGN
- 33000 .....................................UCB, SE, model_noise == EGN
- 33500 ......................................UCB, EP, model_noise == EGN
- 34000 ................LGCP+1/30, UCB, SE, model_noise == EGN
- 34500 .................LGCP+1/30, UCB, EP, model_noise == EGN
- 36500 ............UCB, SE, mapWithNoise, model_noise == EGN
- rand/ ..........................................................Random grid search
