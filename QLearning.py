import time
import gymnasium as gym
import numpy as np
import random
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))
#qlerning.py
learning_rate = 0.1 #alpha
discount_factor = 0.99 #gamma
epsilon = 1.0 #exploration rate
episilon_decay = 0.001 #decay rate per episode 
min_episilon = 0.01 #minimum exploration rate
total_episodes = 1000 #number of episodes
max_steps = 100 #maximum steps per episode

for  episode in range(total_episodes):
    state,info=env.reset()
    done = False 
    while not done:
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()#explore
        else:
            action = np.argmax(q_table[state])#exploit
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        best_next_q = np.max(q_table[next_state])
        q_table[state][action] += learning_rate * (reward + discount_factor * best_next_q - q_table[state][action])
        state = next_state
    epsilon = max(min_episilon, epsilon * episilon_decay)
print("Traning finished. Optimal Q_table generated.")
env.close()
            
num_test_episodes = 5
for episode in range(num_test_episodes):
    state,info = env.reset()
    done = False
    print(f"Episode {episode + 1}")

    while not done:
        action = np.argmax(q_table[state])
        state, reward,terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.5)
        
    if reward==1:
        print("Success! Reached the Goal")
    else:
        print("Failed! Fell into the hole")
env.close()