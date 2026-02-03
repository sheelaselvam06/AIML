import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import torch.distributions as dist

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        feature = self.shared(state)
        action_probs = self.actor(feature)
        state_value = self.critic(feature)
        return action_probs, state_value

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
actor_optimizer = optim.Adam(model.actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(model.critic.parameters(), lr=0.001)

episodes = 100
gamma = 0.99

for episode in range(episodes):
    state, _ = env.reset()
    log_probs, values, rewards = [], [], []
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        action_probs, state_value = model(state_tensor)
        
        # Sample action
        dist_obj = dist.Categorical(action_probs)
        action = dist_obj.sample()
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        # Store values
        log_probs.append(dist_obj.log_prob(action))
        values.append(state_value)
        rewards.append(reward)
        
        state = next_state
    
    # Calculate returns and advantages
    returns = []
    G = 0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.cat(values)
    advantages = returns - values
    
    # Calculate losses
    actor_loss = -torch.sum(torch.stack(log_probs) * advantages.detach())
    critic_loss = F.mse_loss(values, returns)
    
    # Update networks
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    actor_loss.backward(retain_graph=True)
    critic_loss.backward()
    
    actor_optimizer.step()
    critic_optimizer.step()
    
    print(f"Episode {episode}, Total Reward: {sum(rewards)}")

env.close()