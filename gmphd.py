# import os 

# os.system("bash install_packages.sh")
"""
Multi-Target Tracking: Probability Hypothesis Density Filter (MTT: PHD)

Multi-Target Tracking (MTT): estimation at each timestep of states based on sequence of cluttered sets and noise 
                                as well as an estimation of the number of targets present. 

Uses collection of targets and measurements as random finite set then applies Probability Hypothesis Density Filter

Probability Density Filter: jointly estimates number of targets relatie to time and determines its sequence. 
                            Operates on single-target state space. 

Implementation with Gaussian: similar to Kalman filter. Uses the Gaussian to propagate the PDF with Bayes Recursion.
- Advantage is that it allows state estimates to be extracted from the posterior intensity

Note: 
- Uses Monte Carlo to generate sequential data to propafate posterior intensity.
- PHD requires less  copmutation than Multuple target recursion

Assumptions of Linear Gaussian Multiple-Target Model (from paper)
A1: Each target evolves and generates observations independent of one another
A2: Clutter is Poisson and independent of target originated measurements 
A3: Predicted multiple-target RFS governed by p_{k|k-1} in Poission
A4: Each target follows a linear Gaussian model and the sensor has a linear Gaussian dynamical model and the sensors has a linear Gaussian measurement model
A5: The survival and detection probabilities are state independent 
A6: The intensities of the birth and spawn RFS are Gaussian mixtures

Proposition of PHD show how the Gaussian components of the posterior intensity are analytically propagated to the next time

Based on the pseudocode here: https://ieeexplore-ieee-org.eres.library.manoa.hawaii.edu/document/1710358
"""
import numpy as np

"""
defintion of variables relative to paper

w = weights
J = num components
m = position [x, velocity_x, y, velocity_y]
z = measurements 
P = covariance matrix
"""

class mtt_phd:
    """
    declare all variables use as basis
    """
    """                   w          m      P           J              z"""
    def __init__(self, weights, position, p_cov, num_components, measurement):
        """w"""
        self.weights = weights 
        """m"""
        self.position = position
        """P"""
        self.birth_conv_matrix = p_cov
        """J"""
        self.n_component = num_components
        """z"""
        self.measurement = measurement
    
    """
    step 1
    prediction for birth targets

    creates possible location and uncertainties of targets based on where 
    they were previously and how expected to move

    args:

    weights 
    position
    covariance
    n_components


    return: 
    updated weights, position, and covariance
    """
    def predict_birth(self): 
        i = 0
        for j in range(self.n_component):
            print(j)

    
    """
    step 2
    prediction for existing targets

    creates other targets that have the potential to appear in the scene
    Note: did not show up in step 1

    args
    n_components, 
    weights, 
    position


    """
    def predict_exist():
        i = 0
    
    """
    step 3
    construction of PHD update components

    consider and combines all predicted targets overall

    args: 
    n_components, 
    covariance

    """
    def phd_components_update(): 
        i = 0
    
    """
    step 4 
    update

    check how the predicted targets compare to the target measurements

    args
    n_components, 
    weights, 
    position, 
    covariance

    """

    def update(): 
         i = 0
    
    """
    step pruning 

    retrains the best components (e.g, removes inisghnificant components, 
    merge similar components and limit total components to n_components)

    weights, 
    position, 
    covariance, 
    n_components

    """
    def prune_alg(): 
        i = 0
    
    """
    step 5
    output of doing PHD filter
s
    returns a discrete set of the estimated positiosn of targets at each time step

    args

    weights, 
    position, 
    covariance, 
    components
    """

    def return_findings(): 
        i = 0


def main(): 
    covariance = np.load("data_in_progress/covariance.npy")
    print(covariance)
    measurement = np.load("data_in_progress/measurements.npy")
    print(measurement)
    weights = np.load("data_in_progress/weights.npy")
    print(weights)
    obj = mtt_phd(weights=3, position=3, p_cov=3, num_components=3, measurement=3)
    print("I am declared")
    obj.predict_birth()
    print("this is the main function")


if __name__ == "__main__": 
    main()