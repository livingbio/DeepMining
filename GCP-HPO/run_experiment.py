# Author: Sebastien Dubois 
#     for ALFA Group, CSAIL, MIT

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
import sys
import os
from sklearn.neighbors import NearestNeighbors
from smart_sampling import smartSampling

def runExperiment(first_exp,
                  n_exp,
                  parameter_bounds,
                  model = 'GCP',
                  n_random_init = 10,
                  n_smart_iter = 10,
                  n_candidates=500,
                  corr_kernel='squared_exponential',
                  acquisition_function = 'MaxUpperBound',
                  n_clusters = 1,
                  cluster_evol = 'constant',
                  GCP_mapWithNoise=False,
                  GCP_useAllNoisyY=False,
                  model_noise=None):
  
  last_exp = first_exp + n_exp
  print 'Run experiment',first_exp,'to',last_exp

  # Load data
  output = []
  f =open(("scoring_function/output.csv"),'r')
  for l in f:
      l = l[1:-3]
      string_l = l.split(',')
      output.append( [ float(i) for i in string_l] )
  f.close()
  print 'Loaded output file,',len(output),'rows'

  params = np.genfromtxt(("scoring_function/params.csv"),delimiter=',')
  print 'Loaded parameters file, shape :',params.shape

  KNN = NearestNeighbors()
  KNN.fit(params)
  # KNN.kneighbors(p,1,return_distance=False)[0]

  # function that retrieves a performance evaluation from the stored results
  def get_cv_res(p):
      idx = KNN.kneighbors(p,1,return_distance=False)[0]
      all_o = output[idx]
      r = np.random.randint(len(all_o)/5)
      return all_o[(5*r):(5*r+5)]


  ###  Run experiment  ### 

  for n_exp in range(first_exp,last_exp):
      print ' ****   Run exp',n_exp,'  ****'
      ### set directory
      if not os.path.exists("exp_results/exp"+str(n_exp)):
          os.mkdir("exp_results/exp"+str(n_exp))
      else:
          print('Warning : directory already exists')

      all_parameters,all_raw_outputs,all_mean_outputs, all_std_outputs, all_param_path = \
          smartSampling(n_smart_iter,parameter_bounds,get_cv_res,isInt=True,
                        corr_kernel = corr_kernel ,
                        GCP_mapWithNoise=GCP_mapWithNoise,
                        GCP_useAllNoisyY=GCP_useAllNoisyY,
                        model_noise = model_noise,
                        model = model, 
                        n_candidates=n_candidates,
                        n_random_init=n_random_init, 
                        n_clusters=n_clusters,
                        cluster_evol = cluster_evol,
                        verbose=True,
                        acquisition_function = acquisition_function,
                        detailed_res = True)

      ## save experiment's data
      for i in range(len(all_raw_outputs)):
          f =open(("exp_results/exp"+str(n_exp)+"/output_"+str(i)+".csv"),'w')
          for line in all_raw_outputs[i]:
              print>>f,line
          f.close()
          np.savetxt(("exp_results/exp"+str(n_exp)+"/param_"+str(i)+".csv"),all_parameters[i], delimiter=",")
          np.savetxt(("exp_results/exp"+str(n_exp)+"/param_path_"+str(i)+".csv"),all_param_path[i], delimiter=",")

      print ' ****   End experiment',n_exp,'  ****\n'
