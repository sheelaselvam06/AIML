import numpy as np
import gymnasium as gym
import torch
from torch.distributions import Categorical
probs = torch.tensor([0.2,0.6,0.1,0.1])
dist = Categorical(probs)

ls = []
for k in range(10):
    action = dist.sample()
    ls.append(action.item())

print(ls)

log_prob = dist.log_prob(action)
print(log_prob)

