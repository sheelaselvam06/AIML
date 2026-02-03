# FastApi/ml/AdaBoost.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless-friendly
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Reproducible
np.random.seed(101)

# --- small m vs n example ---
m = np.array([10, 50, 30, 40, 20])
n = np.array([40, 0, 15, 70, 35])
plt.figure()
plt.plot(m, n, "-o", label="connected")
plt.scatter(m, n, c="k")
plt.xlabel("m")
plt.ylabel("n")
plt.title("m vs n")
plt.legend()
plt.savefig("m_vs_n.png")
plt.close()

# --- synthetic regression data ---
x = np.linspace(2, 8, 51)
fx = np.sin(x)
y = fx + np.random.normal(0, 0.6, x.size)
Y = np.round(y, 2)           # rounded noisy targets
X = x.reshape(-1, 1)         # feature matrix

# Save function + noisy scatter
plt.figure()
plt.plot(x, fx, label="sin(x)")
plt.scatter(x, Y, s=20, c="C1", label="noisy (rounded)")
plt.legend()
plt.title("function and noisy samples")
plt.savefig("function_and_noisy.png")
plt.close()

# --- train / test split ---
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=41)

# Convert to DataFrame and sort by x for nicer plotting
df_train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))), columns=["x", "y"]).sort_values("x")
df_test = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))), columns=["x", "y"]).sort_values("x")

# Dense grid for smooth prediction curves
X_plot = np.linspace(x.min(), x.max(), 400).reshape(-1, 1)

def safe_mape(y_true, y_pred):
    # sklearn's MAPE raises if y_true has zeros; handle defensively
    try:
        return mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        # fallback to numpy implementation (with epsilon)
        eps = 1e-8
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))

def evaluate_and_plot(model, name, save_prefix):
    model.fit(df_train[["x"]].values, df_train["y"].values)
    yp_train = model.predict(df_train[["x"]].values)
    yp_test = model.predict(df_test[["x"]].values)
    yp_plot = model.predict(X_plot)

    mse_train = mean_squared_error(df_train["y"].values, yp_train)
    mse_test = mean_squared_error(df_test["y"].values, yp_test)
    r2_train = r2_score(df_train["y"].values, yp_train)
    r2_test = r2_score(df_test["y"].values, yp_test)
    mape_train = safe_mape(df_train["y"].values, yp_train)
    mape_test = safe_mape(df_test["y"].values, yp_test)

    print(f"{name:25s} | Train MSE: {mse_train:.4f}  Test MSE: {mse_test:.4f}  Train R2: {r2_train:.4f}  Test R2: {r2_test:.4f}  Train MAPE: {mape_train:.4f}  Test MAPE: {mape_test:.4f}")

    plt.figure(figsize=(8, 4))
    plt.scatter(df_train["x"], df_train["y"], label="train", alpha=0.8)
    plt.scatter(df_test["x"], df_test["y"], label="test", alpha=0.8)
    plt.plot(X_plot.ravel(), yp_plot, ":", label=f"{name} prediction")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(save_prefix)
    plt.savefig(f"{save_prefix}.png")
    plt.close()

# --- Decision Trees ---
evaluate_and_plot(DecisionTreeRegressor(random_state=0), "DecisionTree (full)", "dt_full")
evaluate_and_plot(DecisionTreeRegressor(max_depth=3, random_state=0), "DecisionTree (max_depth=3)", "dt_d3")
evaluate_and_plot(DecisionTreeRegressor(max_depth=5, random_state=0), "DecisionTree (max_depth=5)", "dt_d5")

# Save the structure of a chosen tree (e.g., max_depth=5)
chosen_tree = DecisionTreeRegressor(max_depth=5, random_state=0)
chosen_tree.fit(df_train[["x"]].values, df_train["y"].values)
plt.figure(figsize=(10, 6))
plot_tree(chosen_tree, feature_names=["x"], filled=True)
plt.savefig("decision_tree_plot.png")
plt.close()

# --- Random Forests ---
evaluate_and_plot(RandomForestRegressor(n_estimators=100, random_state=0), "RandomForest (100)", "rf_100")
evaluate_and_plot(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0), "RandomForest (100, max_depth=5)", "rf_100_d5")

# --- AdaBoost (regressor) with small decision stumps as base learners ---
ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)
evaluate_and_plot(ada, "AdaBoost (stumps)", "ada_boost")

print("All plots saved as PNG in the current working directory.")