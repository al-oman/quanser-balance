import gymnasium as gym
import numpy as np
from collections import deque
import environment  # triggers register()
from utils import get_state_index, setup_csv_file, log_reward_to_csv, save_qtable

env = gym.make("RotPendEnv-v0", max_episode_steps=10_000)
dt = env.unwrapped.dt  # sim step duration
action_repeat = 5

total_time = 10.0  # total simulation time in seconds
total_steps = int(total_time / dt)

gamma = 0.99  # discount factor
alpha = 0.1  # learning rate

n_episodes = 100_000

# intialize q table
action_dim = 5
state_dim = 5 * 5 * 5 * 5
qtable = np.random.rand(state_dim, action_dim)

action_space = np.linspace(-2.0, 2.0, action_dim)  # 5 discrete actions from -2V to +2V

rolling_rewards = deque(maxlen=100)  # track rewards for last 100 episodes
rolling_lengths = deque(maxlen=100)  # track episode lengths for last 100 episodes

# Setup CSV file for logging rewards
csv_file = setup_csv_file()


#======================================= Training Loop ======================================#
for episode in range(n_episodes):
    state, info = env.reset()
    epsilon = max(0.01, 1.0 - episode / (n_episodes * 0.7))
    episode_reward = 0
    sim_steps = 0
    for step in range(total_steps // action_repeat):
        # turn state into qtable index
        state_index = get_state_index(state)

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action_index = np.random.randint(0, len(action_space))
        else:
            action_index = np.argmax(qtable[state_index,:])

        # convert action index to voltage
        voltage = action_space[action_index]

        total_reward = 0
        for _ in range(action_repeat):
        # step the environment
            next_state, reward, terminated, truncated, info = env.step(np.array([voltage]))
            total_reward += reward
            sim_steps += 1
            if terminated or truncated:
                break
        episode_reward += total_reward

        # Update Q-table using Q-learning formula
        next_state_index = get_state_index(next_state)
        if terminated:
            best_next_Q = 0  # no future reward if episode ended
        else:
            best_next_Q = np.max(qtable[next_state_index,:])
        Q = qtable[state_index, action_index]

        # Update Q value
        qtable[state_index, action_index] = (1-alpha) * Q + alpha * (total_reward + gamma * best_next_Q)

        state = next_state

        if terminated or truncated:
            break
    
    rolling_rewards.append(episode_reward)
    rolling_lengths.append(sim_steps)
    if (episode+1) % 100 == 0:
        avg_reward = np.mean(rolling_rewards)
        avg_length = np.mean(rolling_lengths)
        print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}, Average Length: {avg_length:.2f}")
        
        # Log rewards to CSV every 100 episodes
        log_reward_to_csv(csv_file, episode+1, avg_reward, avg_length, epsilon)
    
    # Save Q-table every 10k episodes
    if (episode+1) % 10000 == 0:
        save_qtable(qtable, episode+1)


#======================================= Demo Loop ======================================#
env.close()

env = gym.make("RotPendEnv-v0", render_mode="human", max_episode_steps=10_000)
dt = env.unwrapped.dt  # sim step duration

# Run the enviornment greedily picking action using q table
state, info = env.reset()
for step in range(total_steps // action_repeat):
    state_index = get_state_index(state)
    action_index = np.argmax(qtable[state_index,:])
    voltage = action_space[action_index]

    for _ in range(action_repeat):
        state, reward, terminated, truncated, info = env.step(np.array([voltage]))
        env.render()
        if terminated or truncated:
            break
    # if step % 10 == 0:  # render every 20 steps (500Hz/20 = 25 FPS)
    #     env.render()
        print(reward)

    if terminated or truncated:
        print(f"Episode ended at t={step*dt:.3f}s")
        break

env.close()
