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
H = measurement matrix
"""

class mtt_phd:
    """
    declare all synthetic data
    weights = birth weights (gt)
    position = birth positions (gt)
    p_cov = covariance (gt)
    num_components = number of lines trying to gauge 
    measurement = birth measurement 
        note: measurement ==> what the sensor detects, could be the target or a false positive/ noise/ clutter

    F = state_transition_matrix 
    Q = process_noise_matrix = Q 
    num_steps = total number of points trying to predict per component (should be the same for each??? (need to double check))
    H = measurement_matrix
    R = measurement_noise_covariance
    """
    """                   w          m      P           J              z               F                         Q            num steps       H                 R                   det_prob             clutter_rate,   threshold_weight,    merging_threshold, truncation_threshold  """
    def __init__(self, weights, position, p_cov, num_components, measurement, state_transition_matrix, process_noise_matrix, num_steps, measurement_matrix, measurement_noise, detection_probability, clutter_intensity,  threshold_weight, merging_threshold, truncation_threshold):
        # birth data
        self.birth_weights = weights # w
        self.birth_position = position # m
        self.birth_conv_matrix = p_cov # P
        self.n_component = num_components # number of targets
        self.simulated_measurements = measurement # z 
        self.state_transition_matrix = state_transition_matrix 
        self.process_noise_matrix = process_noise_matrix
        self.num_steps = num_steps
        self.measurement_matrix = measurement_matrix
        self.measurement_noise_covariance = measurement_noise 
        # minimum weight needed to extract values
        self.threshold_weight = threshold_weight
        self.merging_threshold = merging_threshold
        self.truncation_threshold = truncation_threshold # make sure squared

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
        self.clutter_intensity = clutter_intensity

        self.weights_total = []
        self.positions_total = []
        self.covariances_total = []

        # 1 component for gaussian, can expand if doing other tyles of PHD filters
        self.sub_components = 1

        # create the predicted positions
        self.incrementer = 0

        self.state_estimates = []

    
    """
    step 1
    prediction for birth targets

    creates possible location and uncertainties of targets based on where 
    they were previously and how expected to move
    """
    def predict_birth(self): 

        # create the predicted positions
        # i = 0

        # creates the predicted data for each birth component
        for j in range(len(self.birth_weights)):
            self.incrementer+=1
            self.predicted_weights.append(self.birth_weights[j])
            self.predicted_positions.append(self.birth_position[j])
            self.predicted_covariance.append(self.birth_conv_matrix[j])
        
        # print("predicted covariance", self.predicted_covariance)
        
        # excluded the second for loop b/c no spawning 
            # e.g, where spawning appropriate = if randomly have things coming into play (e.g., missle launches from boats)
            # e.g. without spawning == people walking on street

    
    """
    step 2
    prediction for existing targets

    creates other targets that have the potential to appear in the scene
    Note: did not show up in step 1
    """

    def predict_exist(self):
        # similar algorithm to spawning in step 1 but difference is looking at previous weight
        self.surviving_weights = []
        self.surviving_positions = []
        self.surviving_covariances = []

        if len(self.previous_weights) > 0:
         # computes the survival points 
            for j in range(len(self.previous_weights)): # uses num steps as it represents how many targets expect to generate
                for l in range(self.sub_components): # loops only once for GM PHD but gives possibility to expand
                    # survival weights
                    self.incrementer+=1
                    # print("I am in the predict exist, this is the previous weight", self.predicted_weights)
                    surviving_weight = self.prob_survival * self.previous_weights[j]
                    self.surviving_weights.append(surviving_weight)
                    # print("I am in the predict exist")

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
            
            # calculate the measurement and the Kalman prediction
            measurement_predicted = self.measurement_matrix @ position_pred
            innovation_covariance_pred = self.measurement_matrix @ covariance_predicted @ self.measurement_matrix.T + self.measurement_noise_covariance
            kalman_pred = covariance_predicted @ self.measurement_matrix.T @ np.linalg.inv(innovation_covariance_pred)
            posterior_covariance_pred = (np.eye(len(position_pred)) - kalman_pred @ self.measurement_matrix) @ covariance_predicted

            # save variables for future use
            self.predicted_calc_measurement.append(measurement_predicted)
            self.innovation_covariance.append(innovation_covariance_pred)
            self.kalman_gain.append(kalman_pred)
            self.posterior_covariance.append(posterior_covariance_pred)
        
        # BIRTH
        # print("this is the birth covariance", self.birth_conv_matrix)
        # print("this is the predicted weights", self.predicted_weights)
        for j in range(len(self.predicted_weights)):
            # print("this is self.predicted_positions",len(self.predicted_positions))
            position_pred = self.predicted_positions[j]
            covariance_predicted = self.predicted_covariance[j]
            # print("this is covariance_predicted = self.posterior_covariance[j]", covariance_predicted)
            
            measurement_predicted = self.measurement_matrix @ position_pred
            innovation_covariance_pred = self.measurement_matrix @ covariance_predicted @ self.measurement_matrix.T + self.measurement_noise_covariance
            kalman_pred = covariance_predicted @ self.measurement_matrix.T @ np.linalg.inv(innovation_covariance_pred)
            posterior_covariance_pred = (np.eye(len(position_pred)) - kalman_pred @ self.measurement_matrix) @ covariance_predicted

            self.predicted_calc_measurement.append(measurement_predicted)
            self.innovation_covariance.append(innovation_covariance_pred)
            self.kalman_gain.append(kalman_pred)
            self.posterior_covariance.append(posterior_covariance_pred)
        
    """
    Gaussian helper function to get the normals for step 4

    Determines how well the filter makes its predictions
        e.g., if close then low uncertainty, likelihood = high 
        e.g, far away, likelihood = low

    Equation: N(z; n^(j)_k|k-1, S^(j)_k) 
                z = actual measurement
                n = predicted measurement
                S = innovation matrix

    args: 
        residual => actual_measurement - predicted measurement
    """
    def gaussian_likelihood(self, residual, updated_covariance):
        len_residual = len(residual)
        det_innovation = np.linalg.det(updated_covariance)
        if det_innovation <= 0: 
            det_innovation = 1e-10
        norm_constant = 1.0/ (np.power(2 * np.pi, len_residual/2) * np.sqrt(det_innovation))
        exponent = -0.5 * residual.T @ np.linalg.inv(updated_covariance) @ residual
        return norm_constant * np.exp(exponent) 
    
    """
    step 4 
    update

    updates current measuremnts, considers missed detections and measurement updates

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
         # print("this is the predicted weights in update step", self.predicted_weights)
         updated_weights = []
         updated_positions = []
         updated_covariances = []

         all_weights = self.predicted_weights + self.surviving_weights
         all_positions = self.predicted_positions + self.surviving_positions
         all_covariances = self.predicted_covariance + self.surviving_covariances

         # adds in missing detections 
         for j in range(len(all_weights)): 
             weights_missed = (1 - self.detection_probability) * all_weights[j]
             updated_weights.append(weights_missed)
            #  print("i am in the update step with updated weights",updated_weights)
             updated_positions.append(all_positions[j])
             updated_covariances.append(all_covariances[j])
         
         l = 0
        # measurement update --> looks at the intial measurements
         for z in self.current_measurements:
            l = 0
            likelihoods = []

            # compute the likelihood (Gaussian/ normal) that the measurement comes from the predicted target
            for j in range(len(all_weights)): 
                l+=1
                # determines difference between actual measurement and calculated measurement
                residual = z - self.predicted_calc_measurement[j]
                updated_covariance = self.innovation_covariance[j]

                # evaluates if it is part of target
                likelihood = self.gaussian_likelihood(residual, updated_covariance)
                likelihoods.append(likelihood)


            # accounting for true targets and false targets that exist in the clutter
            survivng_rate_weights = [all_weights[w] * likelihoods[w] for w in range(len(all_weights))]
            sum_surviving_rate_weights = sum(survivng_rate_weights)
            #*** kappa --> accounts for all possible explanations of z***
            kappa = self.clutter_intensity + sum_surviving_rate_weights

            # update each predicted component
            for j in range(len(all_weights)):
                residual = z - self.predicted_calc_measurement[j]
                kalman = self.kalman_gain[j]
                position_updated = all_positions[j] + kalman @ residual
                covariance_updated = self.posterior_covariance[j]

                likelihood = likelihoods[j]

                """KEY LINE OF ALGORITHM:
                    Bayes rule for updating the weight of gaussian component given measurement
                    detection probability: probability that a true target is detected
                    surviving weights: prior weight of Gaussian component before considering measurement
                    likelihood: likelihood measurement is true
                    kappa: normalization term --> accounts for all possible explanations (clutter + contributions from all targets)
                """
                weight = (self.detection_probability * all_weights[j]* likelihood) / kappa

                updated_weights.append(weight)
                updated_positions.append(position_updated)
                updated_covariances.append(covariance_updated)
                
         self.updated_weights = updated_weights
         self.updated_positions = updated_positions
         self.updated_covariance = updated_covariances

    
    """
    step pruning 

    retrains the best estimates of the target (e.g, removes inisghnificant positions, 
    merge similar positions and limit total points to steps)

    truncation threshold: determines whether weight is reasonable, (e.g. w > T keep)
    merging threshold: determines how similar predicted targets/components need to be for merging
        Based on the Mahalanobis distance (measures distances between points - including correlated points for multiple variables): 
                Mahalanobis distance = d(i,j) = sqrt((x_b_vector - x_a_vector)^T C^-1 (x_b_vector - x_a_vector))
                                            x_a_vector = point 1
                                            x_b_vector = point 2 
                                            C = sample covariance matrix
                                            source: https://www.statisticshowto.com/mahalanobis-distance/ (has equation with ^0.5)
                                            source: https://www.mathworks.com/help/stats/mahal.html (equation without ^0.5 so decided to keep it as without 0.5)

    maximum allowed gaussians: number of steps (total targets in the overall target)
    """
    def prune_alg(self): 
        l = 0
        maximum_gaussians = self.num_steps

        # goes through all the indices to determine which ones are within the threshold
        indices_keep = [i for i, w in enumerate(self.updated_weights) if w > self.truncation_threshold]

        # finds all the respective values to keep based on the threshold
        # print("this is the updated_weights in prune", self.updated_weights)
        # print("updated_positions in prune", self.updated_positions)
        for index in indices_keep: 
            self.weights_total.append(self.updated_weights[index])
            self.positions_total.append(self.updated_positions[index])
            self.covariances_total.append(self.updated_covariance[index]) 
        
        # print("this is the total_weights in prune", self.weights_total)

        # create variables for merged elements
        merged_weights = []
        merged_positions = []
        merged_covariances = []

        # all indices that need to be considered; set --> ensures each number is unique
        I = set(range(len(self.weights_total)))
        total_merged_targets = 0

        while(I): 
            total_merged_targets+=1
            # finds the largest weight to merge aroudn
            predicted_largest = max(I, key=lambda idx: self.weights_total[idx])

            # L
            componets_closest_to = []

            # finding difference relative to the mahalobis difference and the largest position
            for i in I: 
                # print("updated position in prune", self.updated_positions[i])
                point_1 = np.array([self.positions_total[i][0], self.positions_total[i][1]])
                point_2 = np.array([self.positions_total[predicted_largest][0],self.positions_total[predicted_largest][1]])
                difference_between_positions = point_1 - point_2
                cov_ij = self.covariances_total[i][:2, :2]
                # print("this is the difference between positions", difference_between_positions)
                # print("this is the self.covariance_total",self.covariances_total)
                # print("this is the covariance total relative to time step", self.covariances_total[i])
                mahalobis_difference = difference_between_positions.T @ np.linalg.inv(cov_ij) @ difference_between_positions
                if mahalobis_difference <= self.merging_threshold:
                    componets_closest_to.append(i)
            
            # merging weights; points that are very similar to each other are merged
            weight_summed = sum(self.weights_total[i] for i in componets_closest_to)
            # merges the positions that are similar
            position_summed = sum(self.positions_total[i] * self.weights_total[i] for i in componets_closest_to) / weight_summed

            covariance_summed = np.zeros_like(self.covariances_total[0], dtype=float)

            # determines covariance that are similar and merges them
            for i in componets_closest_to:
            #     difference_between_postions_summed = self.positions_total[i] - position_summed
            #     covariance_summed += self.weights_total[i] * (self.covariances_total[i] + 
            #                                                   np.outer(difference_between_postions_summed, difference_between_postions_summed))
                difference_between_postions_summed = self.positions_total[i] - position_summed  # FULL (4,)
                # print("this is the difference between positions summed shape",difference_between_postions_summed.shape)
                # print("this is the covariance total shape", self.covariances_total[i].shape)
                # print("this is the weights_total[i]", self.weights_total[i])
                covariance_summed += self.weights_total[i] * (
                        self.covariances_total[i] +
                        np.outer(difference_between_postions_summed, difference_between_postions_summed)  # FULL (4x4)
)

            covariance_summed /= weight_summed

            merged_weights.append(weight_summed)
            merged_positions.append(position_summed)
            merged_covariances.append(covariance_summed)
            
            # removed merged/ summed components from the set
            I-=set(componets_closest_to)
        
        # get the amount relative to the total amount of expected targets
        merged_weights_condensed = []
        merged_positions_condensed = []
        merged_covariances_condensed = []

        # print("this is the merged weights",merged_weights)
        
        if len(merged_weights) > maximum_gaussians:
            sorted_indices = np.argsort(merged_weights)[::-1][:maximum_gaussians] # gets it relative to the number of expected
            for i in sorted_indices:
                merged_weights_condensed.append(merged_weights[i])
                merged_covariances_condensed.append(merged_covariances[i])
                merged_positions_condensed.append(merged_positions[i])
        else: 
            merged_weights_condensed = merged_weights
            merged_covariances_condensed = merged_covariances
            merged_positions_condensed = merged_positions
        
        self.updated_covariance = merged_covariances_condensed
        self.updated_positions = merged_positions_condensed
        self.updated_weights = merged_weights_condensed
                
    """
    step 5
    output of doing PHD filter

    determines the final positions being extracted

    returns a discrete set of the estimated positions of targets at each time step
    """
    def return_findings(self):
        # based on weight, determines if fit within threshold
        # print("this is the updated weights in return findings", self.updated_weights)
        for i in range(len(self.updated_weights)):
            # print("this is the weights", self.updated_weights)
            if self.updated_weights[i] >= self.threshold_weight:
                # print("updated weights in prune after threshold",self.updated_weights[i])
                rounded_weight = int(round(self.updated_weights[i]))
                # print("this is the rounded weight",rounded_weight)
                # weight --> how many are possible to be at location
                for _ in range(rounded_weight):
                    self.state_estimates.append(self.updated_positions[i])
        # print("this is the state_estimates",self.state_estimates)
        # extracted positions
        return self.state_estimates
    """
    Runs whole algorithm to determine the predicted targets
    """
    def mtt_phd_whole(self):
        # lists algorithms order
        self.predict_birth()
        self.predict_exist()
        self.phd_components_update()
        self.update()
        self.prune_alg()
        return self.return_findings()
    
    """
    Run the full filter over multiple timesteps to finetune 

    Return: 
        the PHD filter at all the time 
    """
    def full_PHD_filter_run(self): 

        history = []
        for time in range(self.num_steps):
        #for time in range(1):
            self.current_measurements = [m[1] for m in self.simulated_measurements[time]]
           # print("this is the simulated measurements",self.simulated_measurements)
           # print("this is the current measurements", self.current_measurements)
            estimates = self.mtt_phd_whole()
            # print("this is the estimates", estimates)
            history.append(estimates)
            self.previous_positions = self.updated_positions
            self.previous_covariances = self.updated_covariance
            # print("time", time)
            self.previous_weights = self.updated_weights

        return history

def main(): 
    print()

if __name__ == "__main__": 
    main()