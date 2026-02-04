import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
dt = load_iris()
X = dt.data
y = dt.target
 
print("Original shape:", X.shape)
 
# Standardize data for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# PCA with all components (4)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
print("PCA transformed shape:", X_pca.shape)
print("Eigenvalues:", pca.explained_variance_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
 
# Logistic Regression on PCA-transformed data
mod = LogisticRegression()
mod.fit(X_pca, y)
print("Logistic Regression score (4 components):", mod.score(X_pca, y))
 
# Test different numbers of PCA components with train-test split and R2 score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
 
ls = []
r2_scores = []
for k in range(4, 0, -1):
    # PCA on training data
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
   
    # Logistic Regression
    mod = LogisticRegression()
    mod.fit(X_train_pca, y_train)
   
    # Accuracy score
    acc_score = mod.score(X_test_pca, y_test)
    ls.append(acc_score)
   
    # R2 score (for classification, we'll use it as a pseudo-R2 metric)
    y_pred_proba = mod.predict_proba(X_test_pca)
    # Convert to expected values for R2 calculation
    y_pred_expected = np.argmax(y_pred_proba, axis=1)
    r2 = r2_score(y_test, y_pred_expected)
    r2_scores.append(r2)
   
    print(f"Components: {k}, Accuracy: {acc_score:.4f}, R2 Score: {r2:.4f}, Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")
 
print("Accuracy scores:", ls)
print("R2 scores:", r2_scores)
 
# Show feature reduction
print(f"\nFeature Reduction Summary:")
print(f"Original features: {X_scaled.shape[1]}")
for i, k in enumerate(range(4, 0, -1)):
    reduction = X_scaled.shape[1] - k
    reduction_pct = (reduction / X_scaled.shape[1]) * 100
    print(f"Components: {k} | Features removed: {reduction} | Reduction: {reduction_pct:.1f}%")
 
# Graph components against both scores
plt.figure(figsize=(12, 8))
 
# Subplot 1: Scores
plt.subplot(2, 2, 1)
plt.plot(range(4, 0, -1), ls, 'bo-', linewidth=2, markersize=8, label='Accuracy Score')
plt.plot(range(4, 0, -1), r2_scores, 'ro-', linewidth=2, markersize=8, label='R2 Score')
plt.xlabel('Number of PCA Components')
plt.ylabel('Score')
plt.title('PCA Components vs Model Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(4, 0, -1))
 
# Subplot 2: Feature Reduction
plt.subplot(2, 2, 2)
original_features = X_scaled.shape[1]
components = list(range(4, 0, -1))
reductions = [original_features - k for k in components]
reduction_pcts = [(r/original_features)*100 for r in reductions]
 
plt.bar(components, reduction_pcts, color='orange', alpha=0.7)
plt.xlabel('Number of PCA Components')
plt.ylabel('Feature Reduction (%)')
plt.title('Feature Reduction by PCA Components')
plt.xticks(components)
plt.grid(True, alpha=0.3)
 
# Subplot 3: Explained Variance
plt.subplot(2, 2, 3)
explained_variances = []
for k in range(4, 0, -1):
    pca_temp = PCA(n_components=k)
    pca_temp.fit(X_train)
    explained_variances.append(pca_temp.explained_variance_ratio_.sum() * 100)
 
plt.bar(components, explained_variances, color='green', alpha=0.7)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance (%)')
plt.title('Information Preserved')
plt.xticks(components)
plt.grid(True, alpha=0.3)
 
# Subplot 4: Combined view
plt.subplot(2, 2, 4)
plt.plot(components, ls, 'b-', linewidth=2, label='Accuracy')
plt.plot(components, r2_scores, 'r-', linewidth=2, label='R2')
plt.plot(components, explained_variances, 'g--', linewidth=2, label='Explained Variance %')
plt.xlabel('Number of PCA Components')
plt.ylabel('Percentage / Score')
plt.title('Combined Metrics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(components)
 
plt.tight_layout()
plt.savefig('pca_comprehensive.png')
 
 