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

class mtt_phd:
    """
    declare all variables use as basis
    """
    def __init__(self, weights, means, p_cov, target_set):
        self.birth_weights = weights
        self.birth_gaussian_means = means
        self.birth_conv_matrix = p_cov
        # self.measurement_set = z_k
        self.birth_target_set = target_set
    
    """
    step 1
    prediction for birth targets
    """
    def predict_birth(): 
        i = 0
        for j in len():
            print()
    
    """
    step 2
    prediction for existing targets
    """
    def predict_exist():
        i = 0
    
    """
    step 3
    construction of PHD update components
    """
    def phd_components_update(): 
        i = 0
    
    """
    step 4 
    update
    """

    def update(): 
         i = 0
    
    """
    step pruning 

    retrains the best components
    """
    def prune_alg(): 
        i = 0
    
    """
    step 5
    output of doing PHD filter
    """

    def return_findings(): 
        i = 0


def main(): 
    print("this is the main function")


if __name__ == "__main__": 
    main()