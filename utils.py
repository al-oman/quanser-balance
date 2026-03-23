import numpy as np

# intialize q table
action_dim = 5
state_dim = 5 * 5 * 5 * 5
qtable = np.random.rand(state_dim, action_dim)

action_space = np.linspace(-2.0, 2.0, action_dim)  # 5 discrete actions from -2V to +2V

def get_state_index(obs):
    # discretize the continuous state into 5 bins for each of the 4 dimensions

    theta_1_max = 15*np.pi/180
    theta_2_max = 15*np.pi/180
    theta_1_dot_max = 2
    theta_2_dot_max = 2

    bins = [np.linspace(-theta_1_max, theta_1_max, 5),  # theta_1
            np.linspace(-theta_2_max, theta_2_max, 5),  # theta_2
            np.linspace(-theta_1_dot_max, theta_1_dot_max, 5),  # theta_1_dot
            np.linspace(-theta_2_dot_max, theta_2_dot_max, 5)]  # theta_2_dot

    indices = []
    for i in range(4):
        idx = np.digitize(obs[i], bins[i]) - 1  # get bin index
        idx = np.clip(idx, 0, 4)  # ensure index is within bounds
        indices.append(idx)

    # convert multi-dimensional indices to a single index for the q table
    state_index = (indices[0] * (5**3) + 
                   indices[1] * (5**2) + 
                   indices[2] * (5**1) + 
                   indices[3] * (5**0))
    return int(state_index)
