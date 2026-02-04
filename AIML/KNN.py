from sklearn.datasets import make_moons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(noise=0.3, random_state=47)
print(X)
print(y)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('knn1.png')
plt.clf()

xtrain, xtest, ytrain, ytest = train_test_split(X, y,random_state=32)
plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain)
plt.savefig('knn2.png')
plt.clf()

model = SVC(kernel='rbf', C=1.0)

model.fit(xtrain,ytrain)
model.score(xtrain,ytrain)

test=[]
train=[]
for c in [0.1, 1, 10, 100]:
    model = SVC(kernel='rbf', C=c)
    model.fit(xtrain,ytrain)
    train.append(model.score(xtrain,ytrain))
    test.append(model.score(xtest,ytest))
    print(f"C={c}, Train Score: {model.score(xtrain,ytrain)}")
    print(f"C={c}, Test Score: {model.score(xtest,ytest)}")
plt.plot([0.1, 1, 10, 100], train, label='train')
plt.plot([0.1, 1, 10, 100], test, label='test')
plt.xscale('log')
plt.xlabel('C parameter (log scale)')
plt.ylabel('Score')
plt.legend()
plt.savefig('knn3.png')
plt.clf()

mod = SVC(kernel='rbf', C=10)
mod.fit(xtrain,ytrain)
print(mod.score(xtest,ytest))

import mglearn as mg

mg.discrete_scatter(xtrain[:, 0], xtrain[:, 1], ytrain)
mg.discrete_scatter(xtest[:, 0], xtest[:, 1], ytest)
mg.plots.plot_2d_separator(mod, xtrain)
plt.savefig('knn5.png')
plt.clf()
 