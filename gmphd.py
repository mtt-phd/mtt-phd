# imports packages 
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
    weights = birth weights (gt)
    position = birth positions (gt)
    p_cov = covariance (gt)
    num_components = number of lines trying to gauge 
    measurement = birth measurement 

    F = state_transition_matrix 
    Q = process_noise_matrix = Q 
    num_steps = total number of points trying to predict per component (should be the same for each??? (need to double check))
    H = measurement_matrix
    R = measurement_noise_covariance
    """
    """                   w          m      P           J              z               F                         Q             num steps      H            R                   det_prob"""
    def __init__(self, weights, position, p_cov, num_components, measurement, state_transition_matrix, process_noise_matrix, num_steps, measurement_matrix, measurement_noise, detection_probability):
        # birth data
        self.birth_weights = weights # w
        self.birth_position = position # m
        self.birth_conv_matrix = p_cov # P
        self.n_component = num_components # number of targets
        self.birth_measurement = measurement # z 
        self.state_transition_matrix = state_transition_matrix 
        self.process_noise_matrix = process_noise_matrix
        self.num_steps = num_steps
        self.measurement_matrix = measurement_matrix
        self.measurement_noise_covariance = measurement_noise 

        # how many births are defined in the model,
        # each birth is considered as a hypothesis about where the target may be
        self.total_num = len(self.birth_position)

        # j ==> number of gaussian samples considering

        # predicted data
        self.predicted_weights = []
        self.predicted_positions = []
        self.predicted_covariance = []

        # previous data 
        self.previous_weights = []
        self.previous_positions = []
        self.previous_covariances = []
        
        # surviving data
        self.surviving_positions = []
        self.surviving_weights = []
        self.surviving_covariances = []
        self.prob_survival = 0.99 
        self.survival_rate = 0

        self.detection_probability = detection_probability
        # 1 component for gaussian, can expand if doing other tyles of PHD filters
        self.sub_components = 1

        # create the predicted positions
        self.incrementer = 0

    
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
        # i = 0

        # creates the predicted data for each birth component
        for j in range(self.total_num):
            self.incrementer+=1
            self.predicted_weights.append(self.birth_weights[j])
            self.predicted_positions.append(self.birth_position[j])
            self.predicted_covariance.append(self.birth_conv_matrix[j])
        
        # excluded the second for loop b/c no spawning 
            # e.g, where spawning appropriate = if randomly have things coming into play (e.g., missle launches from boats)
            # e.g. without spawning == people walking on street

    
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
        # similar algorithm to spawning in step 1 but difference is looking at previous weight

        if len(self.previous_weights) > 0:
         # computes the survival points 
            for j in range(len(self.previous_weights)): # uses num steps as it represents how many targets expect to generate
                for l in range(self.sub_components): # loops only once for GM PHD but gives possibility to expand
                    # survival weights
                    self.incrementer+=1
                    surviving_weight = self.prob_survival * self.previous_weights[j]
                    self.surviving_weights.append(surviving_weight)

                    # survival position (addition of d is excluded b/c d = 0 (if used in step 1); predicting the position and not spawning)
                    surviving_position = self.state_transition_matrix @ self.previous_positions[j]
                    self.surviving_positions.append(surviving_position)

                    # survival covariance
                    surviving_covariance = (self.state_transition_matrix @ self.previous_covariances[j] @ self.state_transition_matrix.T) + self.process_noise_matrix
                    self.surviving_covariances.append(surviving_covariance)
            self.total_num = self.incrementer
    
    """
    step 3
    construction of PHD update components

    construct an updated Gaussian components using kalman filter
    computes updated weight based on how closely matches the measurement

    Notes: 
        V^(j)_k|k-1 = H_k m^j_k|(k-1) <-- caculation of predicted measurements
        S^(j)_k = R_k + H^j_k|k-1 P^j_k|k-1 H^T_k <-- innovation covariance
        K^j_k = P^j_k|k-1 H^T_k[S^j_k]^-1 <-- kalman gain
        P^(j)_k|k = Q_k-1 + F_k-1 P^j_k-1 F^T_k-1 <-- posterior covariance

    args: 
    n_components, 
    covariance

    """
    def phd_components_update(self): 
        # declare variables to reference in later steps
        self.predicted_calc_measurement = []
        self.innovation_covariance = []
        self.kalman_gain = []
        self.posterior_covariance = [] # used in step 4

        # iterate through the surviving points
        for j in range(len(self.surviving_weights)):
            position_pred = self.surviving_positions[j]
            covariance_predicted = self.surviving_covariances[j]
            
            # calculate the measurement and the kalman prediction
            measurement_predicted = self.measurement_matrix @ position_pred
            innovation_covariance_pred = self.measurement_matrix @ covariance_predicted @ self.measurement_matrix.T + self.measurement_noise_covariance
            kalman_pred = covariance_predicted @ self.measurement_matrix.T @ np.linearalg.inv(innovation_covariance_pred)
            posterior_covariance_pred = (np.eye(len(position_pred)) - kalman_pred @ self.measurement_matrix) @ covariance_predicted

            # save variables for future use
            self.predicted_calc_measurement.append(measurement_predicted)
            self.innovation_covariance.append(innovation_covariance_pred)
            self.kalman_gain.append(kalman_pred)
            self.posterior_covariance.append(posterior_covariance_pred)

    
    """
    step 4 
    update

    updates current measuremnts, considers missed detections and measurement updates

    args
    n_components, 
    weights, 
    position, 
    covariance

    PSEUDO Code: 
        for j in J_k|k-1
            w^(j)_k = (1 - P_D,k) w^(j)_k|k-1 
            m^(j)_k = m^(j)_k|k-1
            P^(j)_k = P^(j)_k|k-1
        l:=0
        for each z in Z_k 
            l+=1
            for j in J_k|k-1
                w_k^(l J_k|k-1 + j) = p_D,k w^(j)_k|k-1 N(z; n^(j)_k|k-1, S^(j)_k)
                m_k^(l J_k|k-1 + j) = m^j_k|k-1 + K^(j)_k (z - n^(j)_k|k-1)
                P_k^(l J_k|k-1 + j) = P^(j)_k|k
            
            w_k^(l J_k|k-1 +j) := w_k^(l J_k|k-1 + j) / ( K_k(z) + sigma w_k^(l J_k|k-1 +i) for j in J_k|k-1
        J = l (J_k|k-1) + J_k|k-1

        output: {w^(i)_k, m^(i)_k, P^(i)_k}

    """
    def update(self): 
         updated_weights = []
         updated_positions = []
         updated_covariances = []

         # adds in missing detections 
         for j in range(len(self.surviving_weights)): 
             weights_missed = (1 - self.detection_probability) * self.surviving_weights[j]
             updated_weights.append(weights_missed)
             updated_positions.append(self.surviving_positions[j])
             updated_covariances.append(self.surviving_covariances[j])
         
         l = 0
        # measurement update --> looks at the intial measurements
         for z in self.birth_measurement:
            l = 0
            likelihood = []
            for j in range(len(self.surviving_weights)): 
                l+=1
                residual = z - self.predicted_calc_measurement[l]
    
    def gausian_helper_function(): 
        return 0

        
    
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