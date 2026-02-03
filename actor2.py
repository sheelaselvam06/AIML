import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create environment with render mode
env = gym.make("Hopper-v4", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("State dimension:", state_dim)
print("Action dimension:", action_dim)

# Reset environment
state, _ = env.reset()

# Run a random policy for a few steps
for _ in range(500):
    action = env.action_space.sample()  # random action
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()  # render the environment
    if terminated or truncated:
        state, _ = env.reset()

env.close()
