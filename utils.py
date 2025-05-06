import numpy as np

def create_state_matrix(dt:float = 1.0, dim:int = 2)->np.ndarray:
    """Creates a state transition matrix for an n-dimensional constant velocity model.

    The matrix is composed of four (dim x dim) blocks:

        [ I   I*dt ]
        [ 0     I  ]

    where:
        - I is the identity matrix
        - 0 is the zero matrix
        - dt is the timestep length

    Args:
        dt (float, optional): Length of a single timestep. Defaults to 1.0.
        dim (int, optional): Number of spatial dimensions. Set to 2 for 2D (x, y) motion, 3 for 3D (x, y, z) motion, etc. Defaults to 2.

    Returns:
        np.ndarray: The state transition matrix.
    """
    return np.vstack((
        np.hstack((np.identity(dim), np.identity(dim) * dt)),
        np.hstack((np.zeros((dim, dim)), np.identity(dim)))
    ))


def create_measurement_matrix(dim:int = 2)->np.ndarray:
    """Creates a measurement matrix for an n-dimensional constant velocity model.

    The matrix H maps the full state to position measurements only:

        [ I  0 ]

    where:
        - I is the identity matrix (measures position)
        - 0 is the zero matrix (ignores velocity)

    Args:
        dim (int, optional): Number of spatial dimensions. Set to 2 for 2D (x, y) motion, 3 for 3D (x, y, z) motion, etc. Defaults to 2.

    Returns:
        np.ndarray: The measurement matrix.
    """
    return np.hstack((np.identity(dim), np.zeros((dim, dim))))

def simulate_motion(F:np.ndarray, Q:np.ndarray, num_steps:int = 10, init_truths:int = 3, birth_prob:float = 0.2, death_prob:float = 0.05, pos_bounds:tuple[float, float] = (-30, 30), vel_bounds:tuple[float, float] = (-2, 2))->dict[int, list[np.ndarray]]:
    """Simulates the motion of multiple objects in an n-dimensional space with birth and death probabilities. A generalized form of the previous method to include birth and death probabilities.   

    Args:
        F (np.ndarray): State transition matrix. Must match the dimensions of the state vector.
        Q (np.ndarray): Process noise covariance matrix. Must match the dimensions of the state vector.
        num_steps (int, optional): Number of timesteps to simulate. Defaults to 10.
        init_truths (int, optional): Number of initial objects. Defaults to 3.
        birth_prob (float, optional): Probability that a new object is born. Defaults to 0.2.
        death_prob (float, optional): Probability that an existing object dies. Defaults to 0.05.
        pos_bounds (tuple[float, float], optional): Bounds for the initial position of objects. Defaults to (-30, 30).
        vel_bounds (tuple[float, float], optional): Bounds for the initial velocity of objects. Defaults to (-2, 2).
        
    Returns:  
        dict[int, list[np.ndarray]]: Dictionary containing the states of each object at each timestep. Each key is the object index (0, 1, 2, ...). Each value is a list of state vectors with shape (4,) representing (x, y, dx, dy). 
    """
    
    all_states = {}
    # initial truths
    for i in range(init_truths):
        state = np.array([
            # initial position (x, y) ~ U(pos) x U(pos)
            *np.random.uniform(*pos_bounds, 2),
            # initial velocity (dx, dy) ~ U(vel) x U(vel)
            *np.random.uniform(*vel_bounds, 2),
        ])

        # initial state at t=0
        all_states[i] = [state.copy()]

    # next id for new objects that are born
    next_id = init_truths

    for step in range(1, num_steps + 1):
        
        for i in all_states.keys():
            # get i's previous (t-1) state
            state = all_states[i][-1]

            # if dead, stay dead
            if state is None:
                # None is used to indicate that the object is dead at this timestep
                all_states[i].append(None)
                continue

            # death
            if np.random.rand() <= death_prob:
                # None is used to indicate that the object is dead at this timestep
                all_states[i].append(None)
                continue

            # propagate state (with noise) for each object
            new_state = F @ state + np.random.multivariate_normal(np.zeros(Q[0].size), Q)

            all_states[i].append(new_state.copy())

        # birth
        if np.random.rand() <= birth_prob:
            state = np.array([
                *np.random.uniform(-30, 30, 2),
                *np.random.uniform(-2, 2, 2),
            ])
            # since we are adding a new object, we need to set its state to None for all previous timesteps
            all_states[next_id] = [None] * step + [state.copy()]

            # increment id for the next new object
            next_id += 1

    return all_states

def simulate_motion_basic(F:np.ndarray, Q:np.ndarray, num_steps:int=10, init_truths:int=3)->dict[int, list[np.ndarray]]:
    """Simulates simple motion of multiple objects in an n-dimensional space.

    Args:
        F (np.ndarray): State transition matrix. Must match the dimensions of the state vector.
        Q (np.ndarray): Process noise covariance matrix. Must match the dimensions of the state vector.
        num_steps (int, optional): Number of timesteps to simulate. Defaults to 10.
        init_truths (int, optional): Number of initial objects. Defaults to 3.

    Returns:  
        dict[int, list[np.ndarray]]: Dictionary containing the states of each object at each timestep. Each key is the object index (0, 1, 2, ...). Each value is a list of state vectors with shape (4,) representing (x, y, dx, dy). 
    """
    all_states = {}
    # initial truths
    for i in range(init_truths):
        state = np.array([
            # initial position (x, y) ~ U(-30, 30) x U(-30, 30)
            *np.random.uniform(-30, 30, 2),
            # initial velocity (dx, dy) ~ U(-2, 2) x U(-2, 2)
            *np.random.uniform(-2, 2, 2),
        ])

        all_states[i] = [state.copy()]

    for step in range(1, num_steps + 1):
        
        # propagate state for each object
        for i in all_states.keys():
            # get previous state
            state = all_states[i][-1]

            # propagate state with noise
            new_state = F @ state + np.random.multivariate_normal(np.zeros(Q[0].size), Q)
            
            all_states[i].append(new_state.copy())

    return all_states

def simulate_measurement(ground_truths:dict[int, list[np.ndarray]], H:np.ndarray, R:np.ndarray, num_steps:int=10, det_prob:float=0.9, clutter_rate:float=3.0, area:tuple[float, float]=(-100,100))->dict[int, list[tuple[int, np.ndarray]]]:
    """Simulates the measurement process for multiple objects in an n-dimensional space.

    Args:
        ground_truths (dict[int, list[np.ndarray]]): 
            Dictionary containing the states of each object at each timestep. Each key is the object index (0, 1, 2, ...). Each value is a list of state vectors with shape (4,) representing (x, y, dx, dy).
            
        H (np.ndarray): 
            Measurement matrix. Must match the dimensions of the state vector.

        R (np.ndarray): 
            Measurement noise covariance matrix. Must match the dimensions of the measurement vector. 

        num_steps (int, optional): 
            Number of timesteps to simulate. Defaults to 10.

        det_prob (float, optional): 
            Probability of correctly detecting an object at each timestep. Defaults to 0.9.

        clutter_rate (float, optional): 
            Rate at which clutter (false measurements) appears in the measurement process, modeled by a Poisson distribution. Defaults to 3.0.

        area (tuple[float, float], optional): 
            Bounds for the clutter positions. Defaults to (-100, 100).

    Returns:
        dict[int, list[tuple[int, np.ndarray]]]: 
            Dictionary containing the measurements at each timestep, where:
                - Each key is the timestep index (0, 1, 2, ...). 
                - Each value is a list of tuples, where each tuple contains
                    - the object index (or None for clutter) 
                    - the corresponding measurement vector.
    """
    
    # all measurements
    all_measurements = {}

    for step in range(num_steps):
        measurements = []

        # real measurement
        for i, states in ground_truths.items():
            # if step exceeds the number of true states, continue
            if step >= len(states):
                continue

            # get the state at the current timestep
            state = states[step]

            # if an item is dead at the current timestep, continue
            if state is None:
                continue
            
            # detect an object with detection probability
            # if the object is detected, add the measurement
            if np.random.rand() <= det_prob:
                measurement = H @ state + np.random.multivariate_normal(np.zeros(R[0].size), R)
                measurements.append((i, measurement.copy()))

        # clutter
        for _ in range(np.random.poisson(clutter_rate)):
            # clutter (x, y) ~ U(-100, 100) x U(-100, 100)
            clutter = np.array([
                *np.random.uniform(*area, 2), 
            ])
            measurements.append((None, clutter.copy()))
        
        # the measurements for this timestep consist of the real measurements (obj_id, pos) and clutter (None, pos)
        all_measurements[step] = measurements

    return all_measurements
