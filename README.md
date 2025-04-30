# Multi Target Tracking: Probability Hypothesis Denisty

This repository includes an explanation and implementation of the Gaussian Mixture Probability Hypothesis Density Filter (GM-PHD filter).

The PHD filter is an approximation of the multi-object Bayes filter for multi-target tracking (MTT). The Bayes filter seeks to compute and propagate the posterior probability density of a target's state, which contains all information about the state at a given time. Extending this to a multi-target scenario requires propagating a multi-object posterior, which quickly becomes computationally intractable due to its high dimensionality and the need to account for all possible data associations.

The PHD filter approximates the density by instead propagating its first order statistical moment (mean), referred to as the prosterior intensity. This intensity represents the expected number of objects in some region of the state space, capturing target density without the need to determine individual densities. The Gaussian Mixture PHD (GM-PHD) filter is an implementation of the PHD filter that assumes linear Gaussian models for motion and measurement.


# Team
- Amanda Nitta
- Jayden Tactay