import numpy as np
import csv
import pickle
from pathlib import Path

# intialize q table
action_dim = 5
state_dim = 5 * 5 * 5 * 5
qtable = np.random.rand(state_dim, action_dim)

v_max = 1.0
action_space = np.linspace(-v_max, v_max, action_dim)  # 5 discrete actions from -v_max to +v_max   

def get_state_index(obs):
    # discretize the continuous state into 5 bins for each of the 4 dimensions

    theta_1_max = 15*np.pi/180
    theta_2_max = 15*np.pi/180
    theta_1_dot_max = 0.5
    theta_2_dot_max = 0.5

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

def plot_rewards(filepath):
    import matplotlib.pyplot as plt
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 100)')
    plt.title('Training Progress')
    plt.show()

def setup_csv_file(filename="training_rewards.csv"):
    """Initialize CSV file with headers for logging training rewards."""
    csv_file = Path(filename)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Avg_Reward", "Avg_Length", "Epsilon"])
    return csv_file

def log_reward_to_csv(csv_file, episode, avg_reward, avg_length, epsilon):
    """Append training metrics to CSV file."""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, avg_reward, avg_length, epsilon])

def save_qtable(qtable, episode):
    """Save Q-table to pickle file."""
    qtable_file = Path(f"qtable_episode_{episode}.pkl")
    with open(qtable_file, 'wb') as f:
        pickle.dump(qtable, f)
    print(f"Saved Q-table to {qtable_file}")