 
import sys
import matplotlib
# Set a non-interactive backend so the script can run without a GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
 
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
 
X, y = make_blobs(n_samples=300, centers=2, random_state=0, cluster_std=1.0)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 
import mglearn as mg
 
# Save a simple scatter of the training data
plt.figure()
mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.title('SVM training scatter')
plt.tight_layout()
plt.savefig('svm_train_scatter.png', dpi=150)
plt.close()
print('Saved svm_train_scatter.png')
 
# Linear SVM separator
svm = SVC(kernel='linear', C=0.01).fit(X_train, y_train)
plt.figure()
mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
mg.plots.plot_2d_separator(svm, X_train)
plt.title('Linear SVM separator')
plt.tight_layout()
plt.savefig('svm_linear_separator.png', dpi=150)
plt.close()
print('Saved svm_linear_separator.png')
 
# Polynomial kernel (degree 3)
mod3 = SVC(kernel='poly', degree=3, C=0.01).fit(X_train, y_train)
plt.figure()
mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
mg.plots.plot_2d_separator(mod3, X_train)
plt.title('Polynomial (deg 3) SVM separator')
plt.tight_layout()
plt.savefig('svm_poly_separator.png', dpi=150)
plt.close()
print('Saved svm_poly_separator.png')
 
 
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
 
y_pred = clf.predict(X_test)
 
print("Classification Report:")
print(classification_report(y_test, y_pred))
 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
 
print("Accuracy Score:", accuracy_score(y_test, y_pred))
 
from sklearn.model_selection import GridSearchCV
 
param_grid={'C':[0.1,1,10,100],
            'gamma':[0.01,0.1,1],
            'kernel':['rbf']}
 
svm= SVC()
 
grid=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)
 
print("Best parameters:",grid.best_params_)
 
 