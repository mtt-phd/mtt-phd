# Multi Target Tracking: Probability Hypothesis Density

This repository includes an explanation and implementation of the Gaussian Mixture Probability Hypothesis Density Filter (GM-PHD filter).

The PHD filter is an approximation of the multi-object Bayes filter for multi-target tracking (MTT). The Bayes filter seeks to compute and propagate the posterior probability density of a target's state, which contains all information about the state at a given time. Extending this to a multi-target scenario requires propagating a multi-object posterior, which quickly becomes computationally intractable due to its high dimensionality and the need to account for all possible data associations.

The PHD filter approximates the density by instead propagating its first order statistical moment (mean), referred to as the prosterior intensity. This intensity represents the expected number of objects in some region of the state space, capturing target density without the need to determine individual densities. The Gaussian Mixture PHD (GM-PHD) filter is an implementation of the PHD filter that assumes linear Gaussian models for motion and measurement.

The file [gmphd.py](gmphd.py) provides an implementation of the GM-PHD filter, adapted from the [original paper](https://ieeexplore.ieee.org/document/1710358) by Vo and Ma (2006).

The file [gmphd.ipynb](gmphd.ipynb) provides a demonstration and explanation of the abstracted class on synthetic multi-object motion and measurement data. While the file [demo.ipynb](demo.ipynb) provides a demonstration excluding the written explanation.

THe file [gmphd_precise.py](gmphd_precise.py) provides a more precise adaptation of the GM-PHD filter, using the variable names from the paper. The file [demo_precise.ipynb](demo_precise.ipynb) is a demonstration that is more precise. 

# Team
- Amanda Nitta
- Jayden Tactay