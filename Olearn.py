import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib

from FastApi.ml.pytorch_gd import learning_rate

matplotlib.use('Agg')

#Initialization environment
env=gym.make("FrozenLake-v1",is_slippery=False)
Q = np.zeros((env.observation_space.n,env.action_space.n))

#modle free = lack of knowledge 
#model based = transition probabilities 
#continuous observation space
alpha=0.1
gamma=0.99
epsilon=1
episilon_max=1.0
episilon_min=0.01
episilon_decay=0.99
decay_rate=0.005
num_episodes=1000
threshold=1e-6

for episode in range(num_episodes):
    state,_=env.reset()
    max_delta=0
    while True:
    # select action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()# Explore
        else:
            action = np.argmax(Q[state])# Exploit
        next_state, reward, done, _,info = env.step(action)
        old_value = Q[state, action]
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        max_delta = max(max_delta, abs(Q[state][action] - Q[state][action]- old_value))
        state = next_state
        if done:
            break
        # decay episilon to reduce exploration over time 
    epsilon = max(episilon_min, epsilon * 0.9)
#check for convergence
    if max_delta < threshold and epsilon < 0.05: # ensure epsilon is low
        print(f"Policy converged after {episode } episodes")
        break