# ...existing code...
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# reproducible
np.random.seed(101)

# simple m/n plot example
m = np.array([10, 50, 30, 40, 20])
n = np.array([40, 0, 15, 70, 35])
plt.figure()
plt.plot(m, n, '-o')
plt.title("m vs n (connected in order)")
plt.xlabel("m")
plt.ylabel("n")
plt.savefig("m_vs_n.png")
plt.close()

# generate synthetic data
x = np.linspace(2, 8, 51)
fx = np.sin(x)
y = fx + np.random.normal(0, 0.6, x.size)
Y = np.round(y, 2)          # rounded targets
X = x.reshape(-1, 1)

# plot clean function and noisy samples
plt.figure()
plt.plot(x, fx, label="sin(x)")
plt.scatter(x, Y, s=20, c="C1", label="noisy")
plt.legend()
plt.savefig("function_and_noisy.png")
plt.close()

# train/test split (keep X as 2D)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=41)

# convert to DataFrame and sort by x for nicer plotting
df_train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))), columns=["x", "y"]).sort_values("x")
df_test = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))), columns=["x", "y"]).sort_values("x")

# dense grid for plotting model curves
X_plot = np.linspace(x.min(), x.max(), 400).reshape(-1, 1)

def evaluate_and_plot(model, name, save_prefix):
    model.fit(df_train[["x"]].values, df_train["y"].values)
    yp_train = model.predict(df_train[["x"]].values)
    yp_test = model.predict(df_test[["x"]].values)
    yp_plot = model.predict(X_plot)

    mse_train = mean_squared_error(df_train["y"].values, yp_train)
    mse_test = mean_squared_error(df_test["y"].values, yp_test)
    r2_train = r2_score(df_train["y"].values, yp_train)
    r2_test = r2_score(df_test["y"].values, yp_test)

    # MAPE can blow up if y contains zeros; handle safely
    try:
        mape_train = mean_absolute_percentage_error(df_train["y"].values, yp_train)
    except Exception:
        mape_train = np.nan
    try:
        mape_test = mean_absolute_percentage_error(df_test["y"].values, yp_test)
    except Exception:
        mape_test = np.nan

    print(f"{name}  Train MSE={mse_train:.4f}  Test MSE={mse_test:.4f}  Train R2={r2_train:.4f}  Test R2={r2_test:.4f}  Train MAPE={mape_train:.4f}  Test MAPE={mape_test:.4f}")

    plt.figure(figsize=(8, 4))
    plt.scatter(df_train["x"], df_train["y"], label="train", alpha=0.8)
    plt.scatter(df_test["x"], df_test["y"], label="test", alpha=0.8)
    plt.plot(X_plot.ravel(), yp_plot, ":", label=f"{name} prediction")  
    plt.legend()
    plt.title(save_prefix)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{save_prefix}.png")
    plt.close()

# Decision tree: full, max_depth=3, max_depth=5
evaluate_and_plot(DecisionTreeRegressor(random_state=0), "DT (full)", "dt_full")
evaluate_and_plot(DecisionTreeRegressor(max_depth=3, random_state=0), "DT (max_depth=3)", "dt_d3")
evaluate_and_plot(DecisionTreeRegressor(max_depth=5, random_state=0), "DT (max_depth=5)", "dt_d5")

# Plot the fitted tree for a chosen tree (e.g., max_depth=5)
chosen_tree = DecisionTreeRegressor(max_depth=5, random_state=0)
chosen_tree.fit(df_train[["x"]].values, df_train["y"].values)
plt.figure(figsize=(12, 6))
plot_tree(chosen_tree, feature_names=["x"], filled=True)
plt.savefig("decision_tree_plot.png")
plt.close()

# Random forest: default and constrained depth
evaluate_and_plot(RandomForestRegressor(n_estimators=100, random_state=0), "RF (n=100)", "rf_100")
evaluate_and_plot(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0), "RF (n=100,max_depth=5)", "rf_100_d5")
# ...existing code...