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

Other resources: 
   Gaussian PHD Filter https://stonesoup.readthedocs.io/en/v0.1b7/auto_tutorials/11_GMPHDTutorial.html
   Data Association for mulit-target tracking https://stonesoup.readthedocs.io/en/v0.1b9/auto_tutorials/06_DataAssociation-MultiTargetTutorial.html 

   Another example of PHD filter implementeed https://github.com/Agarciafernandez/MTT/blob/master/PHD%20filter/GMCPHD_filter.m
"""
import numpy as np
import pandas as pd

"""
defintion of variables relative to paper

w = weights
J = number of gaussian possibilties in model
m = position [x, velocity_x, y, velocity_y]
z = measurements 
P = covariance matrix
F = state transition matrix 
Q = process noise matrix
"""

class mtt_phd:
    """
    declare all synthetic data
    """
    """                   w          m      P           J              z               F                         Q """
    def __init__(self, weights, position, p_cov, num_components, measurement, state_transition_matrix, process_noise_matrix):
        # birth data
        self.birth_weights = weights # w
        self.birth_position = position # m
        self.birth_conv_matrix = p_cov # P
        self.n_component = num_components # number of targets
        self.birth_measurement = measurement # z 
        self.state_transition_matrix = state_transition_matrix
        self.process_noise_matrix = process_noise_matrix

        # how many births are defined in the model,
        # each birth is considered as a hypothesis about where the target may be
        self.birth_num = len(self.birth_position)

        # j ==> number of gaussian samples considering

        # predicted data
        self.predicted_weights = []
        self.predicted_positions = []
        self.predicted_covariance = []

    
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

        # create the predicted positions
        i = 0

        # creates the predicted data for each birth component
        for j in range(len(self.birth_num)):
            i+=1
            self.predicted_weights.append(self.birth_weights[i])
            self.predicted_positions.append(self.birth_position[i])
            self.predicted_covariance.append(self.birth_conv_matrix[i])
        
        for j in range(len(self.birth_num)):
            for l in range(len(self.birth_position)): 
                print(l)

    
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
    def predict_exist(self):
        i = 0
        for j in range(self.n_components): 
            print(j)

    
    """
    step 3
    construction of PHD update components

    consider and combines all predicted targets overall

    args: 
    n_components, 
    covariance

    """
    def phd_components_update(self): 
        i = 0
        for j in range(self.n_component):
            print(j)
    
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

    def update(self): 
         i = 0
         for j in range(self.n_component): 
             print(j)
    
    """
    step pruning 

    retrains the best components (e.g, removes inisghnificant components, 
    merge similar components and limit total components to n_components)

    weights, 
    position, 
    covariance, 
    n_components

    """
    def prune_alg(self, weight): 
        l = 0
        while (self.n_component > 0):
            l+=1
            j = max(weight)

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

    def return_findings(self, weights, position): 
        i = 0
        X = []
        for i in range(self.n_components):
            if weights[i] > 0.5: 
                for j in range(round(weights[i])):
                    X_hat = [X, position]
        return X_hat


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