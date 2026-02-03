
import torch
import torch.nn as nn
import numpy as np

np.random.seed(101)
xx = np.linspace(3,8,11)
yy = xx**2 + 1 + np.random.randn(11)


xx = xx.reshape(-1,1)
xx = torch.FloatTensor(xx)
yy = torch.FloatTensor(yy)
     

from torch.optim import SGD, Adam
     

modl = nn.Sequential(nn.Linear(1,1))
     

# modl(xx)
     

lossfn1 = nn.MSELoss()
# sgd = SGD(modl.parameters(), lr=0.01)
sgd = Adam(modl.parameters(), lr=0.1)
     

for _ in range(100):
  sgd.zero_grad()
  y_pred = modl(xx)
  loss = lossfn1(y_pred, yy)
  loss.backward()
  sgd.step()
  print(loss.item())
ypp = modl(xx).detach().numpy()
     

import matplotlib.pyplot as plt
     

plt.scatter(xx, yy)
plt.scatter(xx, ypp)
