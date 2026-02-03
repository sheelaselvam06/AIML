import numpy as np
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate synthetic classification data (slightly more complex)
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1, flip_y=0.05)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipelines for Scikit-Learn Models
poly_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=3)),
    ('lr', LogisticRegression())
])

mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42))
])

# 4. Cross-Validation (K-Fold)
print("--- Performing 5-Fold Cross-Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results_lr = cross_val_score(poly_lr, X, y, cv=kf)
cv_results_mlp = cross_val_score(mlp, X, y, cv=kf)

print(f"Poly-LR CV Accuracy: {cv_results_lr.mean():.4f} (+/- {cv_results_lr.std() * 2:.4f})")
print(f"MLP CV Accuracy: {cv_results_mlp.mean():.4f} (+/- {cv_results_mlp.std() * 2:.4f})")

# 5. Training final models on training set
print("\n--- Training Final Models ---")
poly_lr.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# 6. PyTorch Logistic Regression
print("\n--- Training PyTorch Model ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.FloatTensor(X_test_scaled)

class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

pt_model = LogisticRegressionPyTorch(2)
criterion = nn.BCELoss()
optimizer = optim.Adam(pt_model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    outputs = pt_model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# 7. Evaluation
def evaluate_model(model, X, y, name, is_pytorch=False):
    if is_pytorch:
        with torch.no_grad():
            y_prob = pt_model(torch.FloatTensor(X)).numpy()
            y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    return y_prob

y_prob_lr = evaluate_model(poly_lr, X_test, y_test, "Scikit-Learn Poly-LR")
y_prob_mlp = evaluate_model(mlp, X_test, y_test, "Scikit-Learn MLP")
y_prob_pt = evaluate_model(pt_model, X_test_scaled, y_test, "PyTorch LR", is_pytorch=True)

# 8. Visualization
def plot_results(models_data, X, y):
    fig, axes = plt.subplots(1, len(models_data), figsize=(18, 5))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for i, (name, model, is_pytorch) in enumerate(models_data):
        if is_pytorch:
            with torch.no_grad():
                grid_scaled = scaler.transform(grid)
                Z = pt_model(torch.FloatTensor(grid_scaled)).numpy()
                Z = (Z > 0.5).astype(float)
        else:
            Z = model.predict(grid)
        
        Z = Z.reshape(xx.shape)
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=20)
        axes[i].set_title(name)
    
    plt.tight_layout()
    plt.savefig("ml/decision_boundaries.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, prob, _ in [("Poly-LR", y_prob_lr, False), ("MLP", y_prob_mlp, False), ("PyTorch LR", y_prob_pt, True)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig("ml/roc_curves.png")
    plt.close()
    print("\nSaved plots to ml/decision_boundaries.png and ml/roc_curves.png")

models_data = [ ("Poly-LR (sk)", poly_lr, False), ("MLP (sk)", mlp, False), ("LR (pt)", pt_model, True) ]
plot_results(models_data, X_test, y_test)
print("\nDone.")


