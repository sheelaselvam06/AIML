import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

# Create sample dataset
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'Age': np.random.normal(35, 15, n_samples),
    'Salary': np.random.normal(50000, 20000, n_samples),
    'Experience': np.random.exponential(5, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
})

# Create target variable based on some logic
data['Target'] = ((data['Age'] > 30) & 
                  (data['Salary'] > 40000) & 
                  (data['Experience'] > 3)).astype(int)

print("Dataset Shape:", data.shape)
print("\nTarget Distribution:")
print(data['Target'].value_counts())

# Preprocess categorical variables
le_gender = LabelEncoder()
le_education = LabelEncoder()

data['Gender_encoded'] = le_gender.fit_transform(data['Gender'])
data['Education_encoded'] = le_education.fit_transform(data['Education'])

# Features for modeling
feature_cols = ['Age', 'Salary', 'Experience', 'Gender_encoded', 'Education_encoded']
X = data[feature_cols]
y = data['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("LOGISTIC REGRESSION ANALYSIS")
print("="*60)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

print(f"Training Accuracy: {lr_model.score(X_train_scaled, y_train):.3f}")
print(f"Test Accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")

# Get coefficients
lr_coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0],
    'Absolute_Coefficient': np.abs(lr_model.coef_[0])
})

# Sort by absolute coefficient value
lr_coefficients = lr_coefficients.sort_values('Absolute_Coefficient', ascending=False)

print("\nLogistic Regression Coefficients:")
print(lr_coefficients)

# Interpret coefficients
print("\nCoefficient Interpretation:")
for _, row in lr_coefficients.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    if coef > 0:
        print(f"• {feature}: Positive influence (+{coef:.3f}) - Increases probability of Target=1")
    else:
        print(f"• {feature}: Negative influence ({coef:.3f}) - Decreases probability of Target=1")

# Get intercept
print(f"\nIntercept: {lr_model.intercept_[0]:.3f}")
print(f"Base probability (when all features=0): {1/(1+np.exp(-lr_model.intercept_[0])):.3f}")

print("\n" + "="*60)
print("DECISION TREE ANALYSIS")
print("="*60)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_model.fit(X_train, y_train)

print(f"Training Accuracy: {dt_model.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {dt_model.score(X_test, y_test):.3f}")

# Get feature importance
dt_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
})

# Sort by importance
dt_importance = dt_importance.sort_values('Importance', ascending=False)

print("\nDecision Tree Feature Importance:")
print(dt_importance)

print("\nFeature Importance Interpretation:")
for _, row in dt_importance.iterrows():
    feature = row['Feature']
    importance = row['Importance']
    if importance > 0:
        print(f"• {feature}: {importance:.3f} ({importance*100:.1f}% of total importance)")

print("\n" + "="*60)
print("COMPARISON VISUALIZATION")
print("="*60)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Logistic Regression Coefficients
lr_coefficients_sorted = lr_coefficients.sort_values('Coefficient')
ax1.barh(lr_coefficients_sorted['Feature'], lr_coefficients_sorted['Coefficient'], 
         color=['red' if x < 0 else 'green' for x in lr_coefficients_sorted['Coefficient']])
ax1.set_title('Logistic Regression Coefficients')
ax1.set_xlabel('Coefficient Value')
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax1.grid(axis='x', alpha=0.3)

# Decision Tree Feature Importance
dt_importance_sorted = dt_importance.sort_values('Importance')
ax2.barh(dt_importance_sorted['Feature'], dt_importance_sorted['Importance'], color='skyblue')
ax2.set_title('Decision Tree Feature Importance')
ax2.set_xlabel('Importance Score')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_coefficients_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=feature_cols, class_names=['0', '1'], 
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Structure', fontsize=16)
plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization files saved:")
print("• model_coefficients_comparison.png - Side-by-side comparison")
print("• decision_tree_structure.png - Tree structure visualization")

print("\n" + "="*60)
print("KEY DIFFERENCES")
print("="*60)

print("Logistic Regression Coefficients:")
print("• Show direction (positive/negative) and magnitude of influence")
print("• Linear relationship assumed")
print("• Easy to interpret: higher absolute value = more important")
print("• Can be used for probability calculations")

print("\nDecision Tree Feature Importance:")
print("• Show relative importance (0-1 scale)")
print("• Non-linear relationships captured")
print("• Based on how much each feature reduces impurity")
print("• Sum of all importances = 1")

# Find most important feature from both models
lr_most_important = lr_coefficients.iloc[0]['Feature']
dt_most_important = dt_importance.iloc[0]['Feature']

print(f"\nMost Important Feature:")
print(f"• Logistic Regression: {lr_most_important}")
print(f"• Decision Tree: {dt_most_important}")

if lr_most_important == dt_most_important:
    print("✅ Both models agree on the most important feature!")
else:
    print("⚠️ Models disagree on the most important feature")
