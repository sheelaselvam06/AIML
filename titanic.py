import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Emphasis\FastApi\ml\tested.csv")

print("Columns in dataset:", df.columns.tolist())

# Fix column names and select correct columns
dfn = df[['Pclass','Sex','Age','Fare','Survived']].dropna()
print(dfn)

# Create male dummy variable
dfn['male']=pd.get_dummies(dfn['Sex'], drop_first=True).astype(int)
print(dfn)

# Fix the .values() issue and use .values
x=dfn[['Pclass','male','Age','Fare']].values
y=dfn['Survived'].values

# Add constant
X=sm.add_constant(x)
print(X)

# Fix the sm.sm.OLS issue - should be sm.OLS
model=sm.OLS(y,X)
results=model.fit()
print(results.summary())

print("\n" + "="*60)
print("IQR OUTLIER DETECTION AND TREATMENT")
print("="*60)

# IQR Outlier Detection
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers, lower_bound, upper_bound

# Detect outliers in numerical columns
numerical_cols = ['Age', 'Fare']
print("\n--- Outlier Detection ---")

for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(dfn, col)
    print(f"{col}:")
    print(f"  Q1: {dfn[col].quantile(0.25):.2f}")
    print(f"  Q3: {dfn[col].quantile(0.75):.2f}")
    print(f"  IQR: {(dfn[col].quantile(0.75) - dfn[col].quantile(0.25)):.2f}")
    print(f"  Lower Bound: {lower:.2f}")
    print(f"  Upper Bound: {upper:.2f}")
    print(f"  Outliers: {outliers.sum()} ({outliers.sum()/len(dfn)*100:.1f}%)")
    print(f"  Outlier values: {dfn[col][outliers].tolist()}")
    print()

# IQR Treatment: Remove outliers
print("--- Removing Outliers ---")
dfn_clean = dfn.copy()

for col in numerical_cols:
    outliers, lower, upper = detect_outliers_iqr(dfn_clean, col)
    dfn_clean = dfn_clean[~outliers]
    print(f"Removed {outliers.sum()} outliers from {col}")

print(f"Original dataset size: {len(dfn)}")
print(f"After removing outliers: {len(dfn_clean)}")
print(f"Removed {len(dfn) - len(dfn_clean)} rows ({(len(dfn) - len(dfn_clean))/len(dfn)*100:.1f}%)")

# IQR Treatment: Cap outliers (Winsorization)
print("\n--- Capping Outliers (Winsorization) ---")
dfn_capped = dfn.copy()

for col in numerical_cols:
    Q1 = dfn[col].quantile(0.25)
    Q3 = dfn[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers before capping
    outliers_before = ((dfn_capped[col] < lower_bound) | (dfn_capped[col] > upper_bound)).sum()
    
    # Cap the values
    dfn_capped[col] = dfn_capped[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"{col}: Capped {outliers_before} outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\n--- Comparison Statistics ---")
print("Original Fare statistics:")
print(dfn['Fare'].describe())
print("\nCapped Fare statistics:")
print(dfn_capped['Fare'].describe())

# Re-run OLS with cleaned data
print("\n" + "="*60)
print("OLS WITH CLEANED DATA")
print("="*60)

# Prepare cleaned data
x_clean = dfn_clean[['Pclass','male','Age','Fare']].values
y_clean = dfn_clean['Survived'].values
X_clean = sm.add_constant(x_clean)

# Fit model with cleaned data
model_clean = sm.OLS(y_clean, X_clean)
results_clean = model_clean.fit()
print("Results after removing outliers:")
print(results_clean.summary())

# Prepare capped data
x_capped = dfn_capped[['Pclass','male','Age','Fare']].values
y_capped = dfn_capped['Survived'].values
X_capped = sm.add_constant(x_capped)

# Fit model with capped data
model_capped = sm.OLS(y_capped, X_capped)
results_capped = model_capped.fit()
print("\nResults after capping outliers:")
print(results_capped.summary())

# Visual comparison
print("\n--- Generating Visualizations ---")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Original Fare
axes[0, 0].hist(dfn['Fare'], bins=30, alpha=0.7, color='red', edgecolor='black')
axes[0, 0].set_title('Original Fare Distribution')
axes[0, 0].set_xlabel('Fare')

# Cleaned Fare
axes[0, 1].hist(dfn_clean['Fare'], bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0, 1].set_title('Fare After Removing Outliers')
axes[0, 1].set_xlabel('Fare')

# Capped Fare
axes[1, 0].hist(dfn_capped['Fare'], bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].set_title('Fare After Capping Outliers')
axes[1, 0].set_xlabel('Fare')

# Boxplot comparison
box_data = [dfn['Fare'], dfn_clean['Fare'], dfn_capped['Fare']]
labels = ['Original', 'Cleaned', 'Capped']
axes[1, 1].boxplot(box_data, labels=labels)
axes[1, 1].set_title('Fare Boxplot Comparison')
axes[1, 1].set_ylabel('Fare')

plt.tight_layout()
plt.savefig('iqr_outlier_treatment.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualization saved as 'iqr_outlier_treatment.png'")