## Deep Mining ##

This [project](http://hdi-project.github.io/DeepMining/) is part of the Human Data Interaction project at CSAIL, MIT.


### Contributors ###
-----
- [Sebastien Dubois](http://bit.ly/SebastienDubois)



### Overview ###
---------------
The **Deep Mining** project aims at finding the best hyperparameter set for a Machine Learning pipeline. A pipeline example for the [handwritten digit recognition problem](http://yann.lecun.com/exdb/mnist/) is presented below. Some hyperparameters indeed need to be set carefully, as the degree for the polynomial kernel of the SVM. Choosing the value of such hyperparameters can be a very difficult task and this project's goal is to make it much easier.

**This software will test iteratively, and smartly, some hyperparameter sets in order to find as quickly as possible the best ones to achieve the best classification accuracy that a pipeline can offer.**

![Fig2](GCP-HPO/Figures/DeepMining_workflow.png?raw=true)


### Methods ###
---------------
The folder **GCP-HPO** contains all the code implementing the **Gaussian Copula Process (GCP)** and a **hyperparameter optimization (HPO)** technique based on it. Gaussian Copula Process can be seen as an improved version of the Gaussian Process, that does not assume a Gaussian prior for the marginal distributions but lies on a more complex prior. This new technique is proved to outperform GP-based hyperparameter optimization, which is already far better than the randomized search.

A paper explaining the GCP approach as well as the hyperparameter process is currently being written and will be linked here as soon as possible. Please consider citing it if you use this work.
