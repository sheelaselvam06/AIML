import matplotlib
matplotlib.use("Agg")   # headless-friendly
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ------------------------------
# 1. Sample Dataset
# ------------------------------
X, y = make_classification(
    n_samples=800,
    n_features=10,
    n_informative=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ------------------------------
# 2. Initialize Models
# ------------------------------
dt = DecisionTreeClassifier()
svm = SVC(probability=True)
lr = LogisticRegression()

# ------------------------------
# 3. Fit Models
# ------------------------------
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)

# ------------------------------
# 4. Get probabilities
# ------------------------------
dt_prob = dt.predict_proba(X_test)[:, 1]
svm_prob = svm.predict_proba(X_test)[:, 1]
lr_prob = lr.predict_proba(X_test)[:, 1]

# ------------------------------
# 5. ROC & AUC for Decision Tree
# ------------------------------
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
dt_auc = auc(dt_fpr, dt_tpr)

# ------------------------------
# 6. ROC & AUC for SVM
# ------------------------------
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
svm_auc = auc(svm_fpr, svm_tpr)

# ------------------------------
# 7. ROC & AUC for Logistic Regression
# ------------------------------
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
lr_auc = auc(lr_fpr, lr_tpr)

# ------------------------------
# 8. Plot ROC Curves Separately
# ------------------------------

# Decision Tree ROC
plt.figure()
plt.plot(dt_fpr, dt_tpr, label=f"AUC = {dt_auc:.3f}")
plt.title("ROC Curve - Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig('auc1.png')
plt.clf()

# SVM ROC
plt.figure()
plt.plot(svm_fpr, svm_tpr, label=f"AUC = {svm_auc:.3f}", color="green")
plt.title("ROC Curve - SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig('auc2.png')
plt.clf()

# Logistic Regression ROC
plt.figure()
plt.plot(lr_fpr, lr_tpr, label=f"AUC = {lr_auc:.3f}", color="red")
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig('auc3.png')
plt.clf()

# ------------------------------
# 9. Print AUC Values
# ------------------------------
print("\nAUC Scores:")
print(f"Decision Tree AUC:        {dt_auc:.4f}")
print(f"SVM AUC:                  {svm_auc:.4f}")
print(f"Logistic Regression AUC:  {lr_auc:.4f}")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ------------------------------
# 1. Sample Dataset
# ------------------------------
X, y = make_classification(
    n_samples=800,
    n_features=10,
    n_informative=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ------------------------------
# 2. Initialize Models
# ------------------------------
dt = DecisionTreeClassifier()
svm = SVC(probability=True)
lr = LogisticRegression()

# ------------------------------
# 3. Fit Models
# ------------------------------
dt.fit(X_train, y_train)
svm.fit(X_train, y_train)
lr.fit(X_train, y_train)

# ------------------------------
# 4. Get probabilities
# ------------------------------
dt_prob = dt.predict_proba(X_test)[:, 1]
svm_prob = svm.predict_proba(X_test)[:, 1]
lr_prob = lr.predict_proba(X_test)[:, 1]

# ------------------------------
# 5. ROC & AUC for Decision Tree
# ------------------------------
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
dt_auc = auc(dt_fpr, dt_tpr)

# ------------------------------
# 6. ROC & AUC for SVM
# ------------------------------
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
svm_auc = auc(svm_fpr, svm_tpr)

# ------------------------------
# 7. ROC & AUC for Logistic Regression
# ------------------------------
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
lr_auc = auc(lr_fpr, lr_tpr)

# ------------------------------
# 8. Plot ROC Curves Separately
# ------------------------------

# Decision Tree ROC
plt.figure()
plt.plot(dt_fpr, dt_tpr, label=f"AUC = {dt_auc:.3f}")
plt.title("ROC Curve - Decision Tree")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig('auc1.png')
plt.clf()

# SVM ROC
plt.figure()
plt.plot(svm_fpr, svm_tpr, label=f"AUC = {svm_auc:.3f}", color="green")
plt.title("ROC Curve - SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig('auc2.png')
plt.clf()

# Logistic Regression ROC
plt.figure()
plt.plot(lr_fpr, lr_tpr, label=f"AUC = {lr_auc:.3f}", color="red")
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig('auc3.png')
plt.clf()

# ------------------------------
# 9. Print AUC Values
# ------------------------------
print("\nAUC Scores:")
print(f"Decision Tree AUC:        {dt_auc:.4f}")
print(f"SVM AUC:                  {svm_auc:.4f}")
print(f"Logistic Regression AUC:  {lr_auc:.4f}")
