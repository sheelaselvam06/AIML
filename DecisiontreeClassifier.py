from sklearn.datasets import load_iris
import mglearn as mg
import matplotlib
matplotlib.use("Agg")   # headless-friendly
import matplotlib.pyplot as plt

# Load the iris dataset
dt = load_iris()
print(dt.keys())

# Prepare features and target
x = dt.data
y = dt.target

# Create and fit full decision tree
from sklearn.tree import DecisionTreeClassifier
mod = DecisionTreeClassifier()
mod.fit(x, y)
print("Feature importances:", mod.feature_importances_)

# Use only two features for visualization
X = x[:, 2:4]  # Select features 2 and 3

# Create and fit decision tree with 2 features
mod1 = DecisionTreeClassifier()
mod1.fit(X, y)

# Visualize the results
plt.figure(figsize=(12, 4))

# Plot 1: Data points
plt.subplot(1, 3, 1)
mg.discrete_scatter(X[:, 0], X[:, 1], y=y)
plt.title("Data Points")

# Plot 2: Full tree decision boundary
plt.subplot(1, 3, 2)
mg.discrete_scatter(X[:, 0], X[:, 1], y=y)
mg.plots.plot_2d_separator(mod1, X)
plt.title("Full Tree")

# Plot 3: Limited depth tree
plt.subplot(1, 3, 3)
mod2 = DecisionTreeClassifier(max_depth=2)
mod2.fit(X, y)
mg.discrete_scatter(X[:, 0], X[:, 1], y=y)
mg.plots.plot_2d_separator(mod2, X)
plt.title("Max Depth=2")
plt.savefig("decision_tree_iris.png")
plt.tight_layout()
plt.show()