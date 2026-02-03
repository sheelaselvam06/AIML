import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
 
env=gym.make("CartPole-v1")
STATE_SIZE=env.observation_space.shape[0]  # 4
ACTION_SIZE=env.action_space.n  # 2
MEMORY_SIZE=10000
LR = 0.001
EPISODES = 10000
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99
BATCH_SIZE = 32
TARGET_UPDATE = 10
 
memory=deque(maxlen=MEMORY_SIZE)
 
class DQN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DQN,self).__init__()
        self.fc1=nn.Linear(input_dim,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,output_dim)
 
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)
 
#distribution = probability
POLICY_net = DQN(STATE_SIZE, ACTION_SIZE)#policy = expectation
target_net = DQN(STATE_SIZE, ACTION_SIZE)
target_net.load_state_dict(POLICY_net.state_dict()) #copy weight
target_net.eval()
 
optimizer = optim.Adam(POLICY_net.parameters(), lr=LR)
loss_fn=nn.MSELoss()
 
for episode in range(EPISODES):
    state, info = env.reset()
    state=np.array(state,dtype=np.float32)#conert array
    done=False
    total_reward=0
    while not done:
        if np.random.rand()<EPSILON:
            action=env.action_space.sample()
        else:
            with torch.no_grad():
                action=torch.argmax(POLICY_net(torch.tensor(state))).item()
        #take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state=np.array(next_state,dtype=np.float32)
        memory.append((state,action,reward,next_state,done))
        state = next_state
        total_reward += reward
#train only if enough sample in memory
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE) #sample batch from memory
            states,actions,rewards,next_states,dones=zip(*batch)
            states = torch.tensor(np.array(states))
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards,dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(np.array(next_states))
            dones = torch.tensor(dones,dtype=torch.float32).unsqueeze(1)
            #compute Q values
            q_values = POLICY_net(states).gather(1,actions) #predicted val
 
            #compute target Q values using target network
            next_q_values=target_net(next_states).max(1,keepdim=True)[0]
 
            #Q(s,a)=Q(s,a)+alpha*[Reward+gamma*max(Q(s',a'))-Q(s,a)] => bellman eq
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones)) #bellman eq
 
            #compute loss function
            loss = loss_fn(q_values, target_q_values.detach())
 
            #optimize the model
            optimizer.zero_grad() #reset gradients -> clear previous
            loss.backward()
            optimizer.step()
           
    #epsilon decay
    EPSILON = max(EPSILON_MIN, EPSILON_DECAY * EPSILON)
   
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(POLICY_net.state_dict())
   
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON:.3f}")
 
env.close()
#train with sarsa
for ep in range(episodes):
    state = env.reset()
action = choose_action(state)
for step in range(max_steps):
    next_state, reward, done,_= env.step(action)
    next_action = choose_action(next_state)
    
    #sarsa update
    q[state, action] +alpha *(reward + gamma * q[next_state, next_action] - q[state, action])
    state, action = next_state, next_action
    if done:
        break      