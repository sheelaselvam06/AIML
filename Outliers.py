import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
np.random.seed(42)
 
# ---------------- Create synthetic dataset ----------------
n_samples = 100
X = np.random.randn(n_samples, 3)
true_coefs = np.array([1.5, -2.0, 3.0])
y = X @ true_coefs + np.random.normal(0, 1, n_samples)
print(X.shape)
# Introduce deliberate outlier
y_with_outlier = y.copy()
y_with_outlier[0] += 50   # extreme outlier
 
# Wrap into DataFrame for clarity
df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
df["y_clean"] = y
df["y_with_outlier"] = y_with_outlier
 
print("\n===== DATASET SAMPLE =====")
print(df.head(10))   # show first 10 rows
 
# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_with_outlier, test_size=0.2, random_state=42
)
 
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
 
# Print train and test data
print("\n===== TRAIN DATA (first 10 rows) =====")
print(pd.DataFrame(X_train, columns=["X1","X2","X3"]).assign(y=y_train).head(10))
 
print("\n===== TEST DATA (first 10 rows) =====")
print(pd.DataFrame(X_test, columns=["X1","X2","X3"]).assign(y=y_test).head(10))
 
# ---------------- Linear Regression ----------------
lr = LinearRegression().fit(X_train, y_train)
mse_outlier_train = mean_squared_error(y_train, lr.predict(X_train))
mse_outlier_test = mean_squared_error(y_test, lr.predict(X_test))
 
# Linear Regression without outlier (drop first row)
X_clean = X[1:]
y_clean = y_with_outlier[1:]
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)
lr_clean = LinearRegression().fit(X_train_clean, y_train_clean)
mse_clean_train = mean_squared_error(y_train_clean, lr_clean.predict(X_train_clean))
mse_clean_test = mean_squared_error(y_test_clean, lr_clean.predict(X_test_clean))
 
# ---------------- Lasso Regression ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
lasso = Lasso(alpha=0.1).fit(X_train_scaled, y_train)
mse_lasso_train = mean_squared_error(y_train, lasso.predict(X_train_scaled))
mse_lasso_test = mean_squared_error(y_test, lasso.predict(X_test_scaled))
 
# ---------------- Huber Regressor ----------------
huber = HuberRegressor().fit(X_train_scaled, y_train)
mse_huber_train = mean_squared_error(y_train, huber.predict(X_train_scaled))
mse_huber_test = mean_squared_error(y_test, huber.predict(X_test_scaled))
 
# ---------------- Print clear results ----------------
print("\n===== RESULTS =====")
print("True Coefficients:", true_coefs)
 
print("\nLinear Regression (with outlier)")
print("Coefficients:", lr.coef_)
print("Train MSE:", round(mse_outlier_train, 3), "Test MSE:", round(mse_outlier_test, 3))
 
print("\nLinear Regression (without outlier)")
print("Coefficients:", lr_clean.coef_)
print("Train MSE:", round(mse_clean_train, 3), "Test MSE:", round(mse_clean_test, 3))
 
print("\nLasso Regression (with outlier)")
print("Coefficients:", lasso.coef_)
print("Train MSE:", round(mse_lasso_train, 3), "Test MSE:", round(mse_lasso_test, 3))
 
print("\nHuber Regression (with outlier)")
print("Coefficients:", huber.coef_)
print("Train MSE:", round(mse_huber_train, 3), "Test MSE:", round(mse_huber_test, 3))
 
# ---------------- Summary Comparison Table ----------------
results = pd.DataFrame({
    "Model": [
        "Linear Regression (with outlier)",
        "Linear Regression (without outlier)",
        "Lasso Regression (with outlier)",
        "Huber Regression (with outlier)"
    ],
    "Train MSE": [
        round(mse_outlier_train, 3),
        round(mse_clean_train, 3),
        round(mse_lasso_train, 3),
        round(mse_huber_train, 3)
    ],
    "Test MSE": [
        round(mse_outlier_test, 3),
        round(mse_clean_test, 3),
        round(mse_lasso_test, 3),
        round(mse_huber_test, 3)
    ]
})
 
print("\n===== TRAIN vs TEST COMPARISON =====")
print(results)
 
 