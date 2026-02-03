import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
  
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Softmax(dim=-1))
    
    def forward(self, x):
        return self.fc(x)

    def act(self, state):
        probs = self.forward(state)   
        dist_obj = dist.Categorical(probs)
        action = dist_obj.sample()
        return action.item(), dist_obj.log_prob(action)

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n 

policy = PolicyNet(obs_dim, act_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

GAMMA = 0.99

for ep in range(100):
    state, _ = env.reset()
    log_probs, rewards = [], []
    done = False 

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32)
        action, log_prob = policy.act(state_t)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
    
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = []
    for log_prob, Gt in zip(log_probs, returns):
        loss.append(-log_prob * Gt)
    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {ep}, total reward: {sum(rewards)}")

env.close()