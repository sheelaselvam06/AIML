import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# Load data
@st.cache_data
def load_data():
    dt = pd.read_csv(r"C:\Emphasis\FastApi\ml\Heart_disease_cleveland_new.csv")
    return dt

# Train model
@st.cache_resource
def train_model():
    dt = load_data()
    X = dt.drop('target', axis=1)
    y = dt['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, train_acc, test_acc, train_r2, test_r2, X.columns

# Main app
st.title("Heart Disease Prediction")
st.write("Using Decision Tree with Train-Test Split and R2 Score")

# Load data and train model
data = load_data()
model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, train_acc, test_acc, train_r2, test_r2, feature_names = train_model()

# Dataset overview
st.subheader("Dataset Overview")
st.write(f"Dataset Shape: {data.shape}")
st.write("First 5 rows:")
st.dataframe(data.head())

# Target distribution
st.subheader("Target Distribution")
st.write(data['target'].value_counts())

# Model performance
st.subheader("Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.write("**Accuracy Scores**")
    st.write(f"Training Accuracy: {train_acc:.4f}")
    st.write(f"Test Accuracy: {test_acc:.4f}")

with col2:
    st.write("**R2 Scores**")
    st.write(f"Training R2 Score: {train_r2:.4f}")
    st.write(f"Test R2 Score: {test_r2:.4f}")

# Scatter plot
st.subheader("Actual vs Predicted (Scatter Plot)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(y_test)), y_test, alpha=0.6, label='Actual', color='blue')
ax.scatter(range(len(y_test_pred)), y_test_pred, alpha=0.6, label='Predicted', color='red', marker='x')
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Heart Disease (0=No, 1=Yes)')
ax.set_title('Actual vs Predicted Heart Disease')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Decision tree visualization
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, class_names=['No Disease', 'Disease'], filled=True, ax=ax)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
st.dataframe(feature_importance)

# Prediction interface
st.subheader("Make Prediction")
st.write("Enter patient details to predict heart disease:")

# Input fields
user_input = {}
for col in feature_names:
    if col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        # Categorical features
        if col == 'sex':
            user_input[col] = st.selectbox(col, [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        elif col == 'cp':
            user_input[col] = st.selectbox(col, [0, 1, 2, 3], format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'][x])
        elif col == 'fbs':
            user_input[col] = st.selectbox(col, [0, 1], format_func=lambda x: 'False' if x == 0 else 'True')
        elif col == 'restecg':
            user_input[col] = st.selectbox(col, [0, 1, 2], format_func=lambda x: ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'][x])
        elif col == 'exang':
            user_input[col] = st.selectbox(col, [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        elif col == 'slope':
            user_input[col] = st.selectbox(col, [0, 1, 2], format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
        elif col == 'ca':
            user_input[col] = st.selectbox(col, [0, 1, 2, 3, 4], format_func=lambda x: f'{x} vessels')
        elif col == 'thal':
            user_input[col] = st.selectbox(col, [0, 1, 2, 3], format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])
    else:
        # Numerical features
        user_input[col] = st.number_input(col, value=float(data[col].mean()))

# Make prediction button
if st.button("Predict Heart Disease"):
    # Create input dataframe
    input_df = pd.DataFrame([user_input])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    # Display result
    st.subheader("Prediction Result")
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    st.write(f"Result: {result}")
    st.write(f"Probability (No Disease): {probability[0]:.4f}")
    st.write(f"Probability (Disease): {probability[1]:.4f}")

st.write("\n---")
st.write("Note: This is for educational purposes only and should not be used for actual medical diagnosis.")
