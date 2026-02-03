import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
     def __init__(self, state_dim, action_dim):
         super(ActorCritic, self).__init__()
         self.fc = nn.Linear(state_dim, 64)
         self.actor = nn.Linear(64, action_dim)
         self.critic = nn.Linear(64, 1)
     def forward(self, state):
         x = torch.relu(self.fc(state))
         action_probs = torch.softmax(self.actor(x) , dim =-1)
         state_value = self.critic(x)
         return action_probs, state_value

state_dim, action_dim = 4, 2
gamma = 0.99
epsilon = 0.2
lr = 0.001
epochs = 10
policy = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)
# PPO TRAINING LOOP

for _ in range(epochs):
    state = torch.rand(state_dim)
    action_probs,value = policy(state)

    action= torch.multinomial(action_probs, 1).item()
    reward = np.random.rand()
    next_state = torch.rand(state_dim)
    next_value = policy(next_state)[1
    
   




_, next_value = policy(next_state)
advantage = rewards + gamma * next_value - value

new_probs, _ = policy(state)
r_t = new_probs[actions] / action_probs[actions]

loss =-torch.min(r_t * advantage, torch.clamp(r_t, 1-epsilon, 1+epsilon)*advantage)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss.item())

x = torch.tensor([0.5, -1.3, 1.5, 0.7])
print(torch.clamp(x, min=-1.2, max=1.2))
