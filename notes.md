# Multi Target Tracking: Probability Hypothesis Density Filter (MTT: PHD)

## Purpose: 
* estimate the state and amount of obejcts in a scene
* filter using a random finite set 


## Fundamentals
* defined over state-space - D(x) ; x => state space 
* output is a scalar real value -> expected number of targets per unit volume of the state-space x 
* total number of expected objects in scene found by integrating PHD over state space 

* single-object probability denisty function such as Gaussian, Gaussian mixture or set of particles is defined over the mean
* multi-object tracking defined over Poisson, multi-Bernoulli 

* PHD filter recursively estimates the PHD
* dervived using multi-object recursive Bayesian estimators so has 2 steps 1) prediction and 2) correction step 

# Links: 
- https://www.mathworks.com/help/fusion/ug/introduction-to-phd-filter.html 
- 