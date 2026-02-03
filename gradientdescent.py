import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.linspace(3, 9, 11)
y = 2 * x + 0.3 + np.random.randn(len(x))
df = pd.DataFrame({'x':x,'y':y})

x = df['x'].values
y = df['y'].values
     
plt.scatter(x,y)
plt.savefig("scattered.png")     

yp = 0*x + 0

plt.scatter(x,y)
plt.plot(x,yp,'*r:')     
plt.savefig("scattered_modified.png")    
n = 11
      


mse = np.mean((y-yp)**2)
b=0
b = b + (0.05)*mse
yp = w*x + b
plt.scatter(x,y)
plt.plot(x,yp,'*r:')
plt.plot(x,w*x + 0, '*b:')
plt.savefig("scattered_modified.png")    


0


w = 0.0
b = 0.0
lr = 0.01
epochs = 6

plt.figure()
plt.scatter(x, y)

for _ in range(epochs):
    yp = w * x + b
    plt.plot(x, yp)

    deriv_dw = (-2) * np.mean(x * (y - yp))
    deriv_db = (-2) * np.mean(y - yp)

    w -= lr * deriv_dw
    b -= lr * deriv_db

    yp = w * x + b
    plt.plot(x, yp)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression - Gradient Descent")
plt.savefig("gradient_descent.png")

w=0
b=0
plt.scatter(x,y)
for i in range(60):

    yp = w * x + b
    plt.plot(x, yp)
    mse = np.mean((y - yp)**2)

    deriv_dw = (-2) * np.mean(x * (y - yp))
    deriv_db = (-2) * np.mean(y - yp)

    # Update weights and intercept
    w = w + 0.001 * (mse)
    b = b + 0.01 * (mse)


    yp = w*x + b
    plt.plot(x,yp)

    print(mse,w,b)

