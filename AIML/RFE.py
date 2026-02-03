import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
 
dt = load_iris()
X = dt.data
y = dt.target
 
model=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=4)
print(model.fit_transform(X, y))
print(model.ranking_)
 
model=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=4)
newdt=model.fit_transform(X, y)
mod=LogisticRegression()
mod.fit(newdt, y)
print(mod.score(newdt, y))
 
for k in range(4,0,-1):
    model=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=k)
    newdt=model.fit_transform(X, y)
    mod=LogisticRegression()
    mod.fit(newdt, y)
    print(k, mod.score(newdt, y))
 
 
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
 
dt = load_iris()
X = dt.data
y = dt.target
 
#logistic
 
model=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=4)
print(model.fit_transform(X, y))
print(model.ranking_)
 
model=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=4)
newdt=model.fit_transform(X, y)
mod=LogisticRegression()
mod.fit(newdt, y)
print(mod.score(newdt, y))
 
ls=[]
for k in range(4,0,-1):
    model=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=k)
    newdt=model.fit_transform(X, y)
    mod=LogisticRegression()
    mod.fit(newdt, y)
    ls.append(mod.score(newdt, y))
    print(k, mod.score(newdt, y))
 
#graph k against score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
plt.plot(range(4,0,-1), ls)
plt.savefig('rfe1.png')
 
 