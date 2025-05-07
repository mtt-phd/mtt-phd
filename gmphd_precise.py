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
    
    def __init__(self, weights, position, p_cov, num_components, measurement, state_transition_matrix, process_noise_matrix, num_steps, measurement_matrix, measurement_noise, detection_probability, clutter_intensity,  threshold_weight, merging_threshold, truncation_threshold, new_birth_weight, survival_prob, max_components):
        # birth data
        self.birth_weights = weights # w
        self.birth_position = position # m
        self.birth_conv_matrix = p_cov # P
        self.n_component = num_components # number of targets
        self.Z = measurement # z 
        self.F = state_transition_matrix 
        self.Q = process_noise_matrix
        self.H = measurement_matrix
        self.R = measurement_noise 
        self.num_steps = num_steps

        # minimum weight needed to extract values
        self.threshold_weight = threshold_weight
        self.U = merging_threshold
        self.Trunc = truncation_threshold # make sure squared

        self.p_D = detection_probability
        self.kappa = clutter_intensity

        self.new_birth_weight = new_birth_weight  # a moderate weight
        self.new_birth_prob = 0.25

        self.w_pred = []
        self.m_pred = []
        self.P_pred = []

        self.w_prev = []
        self.m_prev = []
        self.P_prev = []

        self.w_k = []
        self.m_k = []
        self.P_k = []

        self.p_S = survival_prob
        self.J_max = max_components

    def predict_birth(self, step):
        # add initial birth prediction components
        if step == 0:
            for j in range(len(self.birth_weights)):
                w = self.birth_weights[j]
                m = self.birth_position[j]
                P = self.birth_conv_matrix[j]

                self.w_pred.append(w)
                self.m_pred.append(m)
                self.P_pred.append(P)
        
        # Dynamic birth
        if np.random.rand() < self.new_birth_prob:
            # Add new birth
                
            new_weight = self.new_birth_weight
            # new_birth_position = np.array([*np.random.uniform(-50, 50, 2), avg_velocity_x, avg_velocity_y])
            new_birth_position = np.array([0] * 4)
            new_birth_covariance = np.diag([1000, 1000, 2, 2])

            self.w_pred.append(new_weight)
            self.m_pred.append(new_birth_position)
            self.P_pred.append(new_birth_covariance)

    def predict_exist(self):
        if not self.w_prev:
            return
        for j in range(len(self.w_prev)):
            w = self.p_S * self.w_prev[j]
            m = self.F @ self.m_prev[j]
            P = self.F @ self.P_prev[j] @ self.F.T + self.Q

            self.w_pred.append(w)
            self.m_pred.append(m)
            self.P_pred.append(P)        

    def construct_components(self):
        self.eta = []
        self.S = []
        self.K = []
        self.P_comp = []
        for j in range(len(self.w_pred)):
            eta = self.H @ self.m_pred[j]
            self.eta.append(eta)
            S = self.H @ self.P_pred[j] @ self.H.T + self.R
            self.S.append(S)
            K = self.P_pred[j] @ self.H.T @ np.linalg.inv(S)
            self.K.append(K)
            P_comp = (np.identity(K.shape[0]) - K @ self.H) @ self.P_pred[j]
            self.P_comp.append(P_comp)

    def gaussian(self, z, eta, S):
        d = z - eta
        exponent = -0.5 * d.T @ np.linalg.inv(S) @ d
        coeff = 1 / np.sqrt((2 * np.pi) ** len(z) * np.linalg.det(S))
        return coeff * np.exp(exponent)
    
    def update(self, Z):
        for j in range(len(self.w_pred)):
            w_k = (1 - self.p_D) * self.w_pred[j]
            m_k = self.m_pred[j]
            P_k = self.P_pred[j]
            self.w_k.append(w_k)
            self.m_k.append(m_k)
            self.P_k.append(P_k)
        
        l = 0
        for z in Z:
            l += 1
            for j in range(len(self.w_pred)):
                w_k = self.p_D * self.w_pred[j] * self.gaussian(z, self.eta[j], self.S[j])
                m_k = self.m_pred[j] + self.K[j] @ (z - self.eta[j])
                P_k = self.P_comp[j]
                self.w_k.append(w_k)
                self.m_k.append(m_k)
                self.P_k.append(P_k)
        
            temp = l * len(self.w_pred)
            for j in range(len(self.w_pred)):
                idx = temp + j
                new_wk = self.w_k[idx] / (self.kappa + sum(self.w_k[temp:]))
                self.w_k[idx] = new_wk
        

    def prune(self):
        l = 0
        indices = [i for i in range(len(self.w_k)) if self.w_k[i] > self.Trunc]
        I = set(indices)

        merged_w = []
        merged_m = []
        merged_P = []
        while I:
            l += 1
            j = max(I, key=lambda idx: self.w_k[idx])
            L = []
            for i in I:
                inv_P = np.linalg.pinv(self.P_k[i])
                mahalanobis = (self.m_k[i] - self.m_k[j]).T @ inv_P @ (self.m_k[i] - self.m_k[j])

                # mahalanobis = (self.m_k[i] - self.m_k[j]) @ np.linalg.inv(self.P_k[i]) @ (self.m_k[i] - self.m_k[j])
                if mahalanobis <= self.U:
                    L.append(i)
            L = set(L)

            w_tilde = sum(self.w_k[i] for i in L)
            m_tilde = sum(self.w_k[i] * self.m_k[i] for i in L) / w_tilde
            P_tilde = sum(self.w_k[i] * (self.P_k[i] + (m_tilde - self.m_k[i]) @ (m_tilde - self.m_k[i]).T) for i in L) / w_tilde
            merged_w.append(w_tilde)
            merged_m.append(m_tilde)
            merged_P.append(P_tilde)
            I -= L

        merged_w = np.array(merged_w)
        merged_m = np.array(merged_m)
        merged_P = np.array(merged_P)
        
        if l > self.J_max:
            indices = np.argsort(merged_w)[::-1]
            merged_w = merged_w[indices]
            merged_m = merged_m[indices]
            merged_P = merged_P[indices]
        
        self.w_k = merged_w.tolist()
        self.m_k = merged_m.tolist()
        self.P_k = merged_P.tolist()

    def extract(self):
        # Extract the states with weights above the threshold
        extracted_states = []
        for i in range(len(self.w_k)):
            if self.w_k[i] >= self.threshold_weight:
                rounded_weight = int(round(self.w_k[i]))
                for _ in range(rounded_weight):
                    extracted_states.append(self.m_k[i])
        return extracted_states

    def run(self):
        history = {}
        for step in range(self.num_steps):
            measurements = [z[1] for z in self.Z[step]]

            self.w_pred = []
            self.m_pred = []
            self.P_pred = []
            self.w_k = []
            self.m_k = []
            self.P_k = []

            self.predict_birth(step)
            self.predict_exist()
            self.construct_components()
            self.update(measurements)
            self.prune()
            states = self.extract()

            history[step] = states

            self.w_prev = self.w_k.copy()
            self.m_prev = self.m_k.copy()
            self.P_prev = self.P_k.copy()
        return history
