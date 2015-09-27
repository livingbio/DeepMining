## Copula-based Hyperparameter Optimization for Machine Learning Pipelines ##

Contributor : [Sebastien Dubois](http://bit.ly/SebastienDubois)


This repository contains all the code implementing the **Gaussian Copula Process (GCP)** and a **hyperparameter optimization** technique based on it.
All the code is in Python and mainly uses Numpy, Scipy and Scikit-learn.


### Python scripts ###
-------------------------------
- gcp.py ..............................................................................The class implementing the n(L)GCP
- GCP_utils.py ..................................................................................Utility functions for n(L)GCP
- smart_sampling.py .........................................................Script to run the optimization process
- sampling_utils.py ....................................................Utility function for the optimization process
- sklearn_utils.py .........................................................................Utility function from Scikit-learn
- run_experiment.py ................................................Script to run several trials on a test instance
- Test/analyze_results.py ..................................Script to compute the Q<sup>1</sup> scores based on a trial
- Test/run_result_analysis.py ...........................................Run analyze_results script and save it
- Test/iterations_needed.py .........Script to compute the iterations needed to reach a given gain
- Test/show_iterations_needed.py ........................................................Display iterationsNeeded



### Instructions ###
-------------------------------
One can easily run a GCP-based hyperparameter optimization process thanks to this code. This is mostly done by the **SmartSearch** object, which iteratively ask to assess the quality of a selected hyperparameter set. This quality should be returned by the **scoring function** which is implemented by the user and depends on the pipeline. This function should return a list of performance estimations, which would usually be either a single estimation or all k-fold cross-validation results.

To run it on a new pipeline, create a folder *newPipeline* in the Test folder, and create a Python script as run_exp.py in CodeTest_SmartSearch.
The SmartSmapling function has many parameters but most of them have default values. Basically the user just has to provide a *scoring_function* and a *parameter_bounds* array (n_parameters,2). The software will try to find the best parameter set within these ranges by iteratively calling the *scoring_function*.

### Examples ###
-------------------------------
This repository contains two tests **CodeTest_GCP** and **CodeTest_SmartSampling** that enable the user to quicly test the GCP and SmartSampling code. The script display_smartSampling enables to simulate a Smart Sampling process while showing the GCP-based predictions and the acquisition functions (see figure below). However this script does not directly use the smartSampling method and thus should not be used for testing purposes, or only after having been modified accordingly.

The **Branin** and **Hartmann 6D** functions are two artificial examples which are standard test instances for optimization processes. Their evaluation is fast so there is no need to store their values in the scoring_function folder. Note that these functions handle floating point so that can be useful for test purposes as the following examples are made with integers.

The two real examples contained in the repository are the **Sentiment Analysis problem** for IMDB reviews (cf. [Kaggle's competition](https://www.kaggle.com/c/word2vec-nlp-tutorial)) in folder Test/Bags_of_Popcorn, and the **Handwritten digits** one from the MNIST database (cf. [Kaggle's competition](https://www.kaggle.com/c/digit-recognizer)) in folder Test/MNIST.
In order to quickly test the optimization process, a lot of off-line computations have already been done and stored in the folders *Test/ProblemName/scoring_function*. This way, the script run_experiments makes it easy to run fast experiments by querying those files, instead of really building the pipeline for each parameter test.


![Fig1](Figures/SmartSampling_example.png?raw=true)
*An example of the Smart Search process. The function to optimize is the blue line, and we start the process with 10 random points for which we know the real value (blue points). At each step, the performance function is modeled by a GCP and predictions are made (red crosses) based on the known data (blue and red points). The cyan zone shows the 95% condifence bounds. At each step the selected point (the one that maximizes the upper confidence bound) is shown in yellow. This point is then added to the known data so that the model becomes more and more accurate.*


### Directory structure ###
-------------------------------
Each test instance follows the same directory structure, and all files are in the folder Test/ProblemName :
- run_test.py : run several trials by setting the configuration for the script run_experiment
- scoring_function/ : the off-line computations stored. params.csv contains the parameters tested, and output.csv the raw outputs given by the scoring function (all the cross-validation estimation). The files *true_score_t_TTT_a_AAA* refer to the Q<sup>1</sup> scores computed with a threshold == TTT and alpha == AAA
- exp_results/expXXX : run_test stores the results in the folder expXXX where XXX is a integer refering to a configuration
- exp_results/transformed_t_TTT_a_AAA/expXXX : the analyzed results from the trial expXXX, computed by run_result_analysis with a threshold == TTT and alpha == AAA. 
- exp_results/transformed_smooth_t_TTT_a_AAA_kKKK_rRRR_bBBB/expXXX : the analyzed results from the trial expXXX with the *smooth* quality function, computed by run_result_analysis with a threshold == TTT, alpha == AAA, using the nearest KKK neighbors, a radius coefficient RRR and beta == BBB. 
- exp_results/iterations_needed/expXXX_YYY_t_TTT_a_AAA : the mean, median, first and third quartiles of the iterations needed to reach a given score gain, over experiments XXX to YYY. The score is actually the true score computed with a threshold == TTT and alpha == AAA.


.

----------------

Notes :

1. The Q score here refers to the quality function of the Deep Mining paper using the discrete mean values (after performing Welch's t-test) and the standard deviation of the multiple performance estimations.

---------------

.
#### Acknowledgments ####
* Many thanks to [Kalyan Veeramachaneni](http://www.kalyanv.org/) who originated this project during my visit at [Alfa Group](http://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/) (CSAIL, MIT), and for all his great advice.
* I would also like to thank Scikit-learn contributors as this code is based on Scikit-learn's GP implementation.
