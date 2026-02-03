import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Survival Analysis", layout="wide")

st.title(" Titanic Survival Analysis")
st.markdown("Analyze which factors most affected survival in the Titanic disaster")

# Load data
@st.cache_data
def load_data():
    # Create sample Titanic-like dataset
    np.random.seed(42)
    n_passengers = 200
    
    return pd.DataFrame({
        'Age': np.random.normal(30, 15, n_passengers),
        'Sex': np.random.choice(['male', 'female'], n_passengers),
        'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.2, 0.3, 0.5]),
        'Fare': np.random.exponential(30, n_passengers),
        'Survived': np.random.choice([0, 1], n_passengers, p=[0.6, 0.4])
    })

test = load_data()

# Ensure realistic age ranges
test['Age'] = np.clip(test['Age'], 0, 80)
test['Fare'] = np.clip(test['Fare'], 0, 500)

# Sidebar for controls
st.sidebar.header("Analysis Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data")
show_age_distribution = st.sidebar.checkbox("Show Age Distribution")
show_survival_charts = st.sidebar.checkbox("Show Survival Charts")

# Main content
if show_raw_data:
    st.subheader(" Raw Dataset")
    st.dataframe(test.head(10))
    st.write(f"Dataset shape: {test.shape}")

# Age categorization function
def age_category(x):
    if x < 16:
        return 'child'
    elif x >= 16 and x < 40:
        return 'middle'
    else:
        return 'old'

# Process age data
age_data = test['Age'].fillna(0).values
age_df = pd.DataFrame(age_data)
age_df.loc[age_df[0] < 16, 'category'] = 'child'
age_df.loc[(age_df[0] >= 16) & (age_df[0] < 40), 'category'] = 'middle'
age_df.loc[age_df[0] >= 40, 'category'] = 'old'
age_df[0] = age_df[0].apply(age_category)

# Add age category to main dataset
test['Age_Category'] = age_df[0]

# Age distribution
if show_age_distribution:
    st.subheader("👥 Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    age_counts = test['Age_Category'].value_counts()
    ax.pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Passenger Age Distribution")
    st.pyplot(fig)

# Survival analysis
st.subheader("🔍 Survival Analysis")

# Calculate survival rates
age_survival = test.groupby('Age_Category')['Survived'].mean()
sex_survival = test.groupby('Sex')['Survived'].mean()
class_survival = test.groupby('Pclass')['Survived'].mean()

# Fare categories
test['Fare_Category'] = pd.cut(test['Fare'].fillna(0), bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
fare_survival = test.groupby('Fare_Category')['Survived'].mean()

# Display survival rates in columns
col1, col2 = st.columns(2)

with col1:
    st.write("**Survival Rate by Age Category**")
    for age_cat, rate in age_survival.items():
        st.write(f"{age_cat}: {rate:.2%}")
    
    st.write("**Survival Rate by Sex**")
    for sex, rate in sex_survival.items():
        st.write(f"{sex}: {rate:.2%}")

with col2:
    st.write("**Survival Rate by Passenger Class**")
    for pclass, rate in class_survival.items():
        st.write(f"Class {pclass}: {rate:.2%}")
    
    st.write("**Survival Rate by Fare Category**")
    for fare_cat, rate in fare_survival.items():
        st.write(f"{fare_cat}: {rate:.2%}")

# Survival charts
if show_survival_charts:
    st.subheader("📈 Survival Visualization")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age survival
    age_survival.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Survival Rate by Age')
    ax1.set_ylabel('Survival Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    # Sex survival
    sex_survival.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Survival Rate by Sex')
    ax2.set_ylabel('Survival Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    # Class survival
    class_survival.plot(kind='bar', ax=ax3, color='lightgreen')
    ax3.set_title('Survival Rate by Class')
    ax3.set_ylabel('Survival Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Fare survival
    fare_survival.plot(kind='bar', ax=ax4, color='gold')
    ax4.set_title('Survival Rate by Fare')
    ax4.set_ylabel('Survival Rate')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# Factor influence analysis
st.subheader(" Factor Influence Ranking")

# Calculate influence
survival_factors = {
    'Age': age_survival.max() - age_survival.min(),
    'Sex': sex_survival.max() - sex_survival.min(),
    'Class': class_survival.max() - class_survival.min(),
    'Fare': fare_survival.max() - fare_survival.min()
}

sorted_factors = sorted(survival_factors.items(), key=lambda x: x[1], reverse=True)

# Display results
col1, col2, col3 = st.columns(3)

with col1:
    st.success(f"**Most Influential:** {sorted_factors[0][0]}")
    st.write(f"Difference: {sorted_factors[0][1]:.3f}")

with col2:
    if len(sorted_factors) > 2:
        st.info(f"**Middle Factor:** {sorted_factors[1][0]}")
        st.write(f"Difference: {sorted_factors[1][1]:.3f}")

with col3:
    st.error(f"**Least Influential:** {sorted_factors[-1][0]}")
    st.write(f"Difference: {sorted_factors[-1][1]:.3f}")

# Complete ranking
st.write("**Complete Ranking (Most to Least Influential):**")
ranking_df = pd.DataFrame(sorted_factors, columns=['Factor', 'Influence Score'])
ranking_df['Influence Score'] = ranking_df['Influence Score'].round(3)
st.dataframe(ranking_df, use_container_width=True)

# Key insights
st.subheader("💡 Key Insights")
most_influential = sorted_factors[0][0]
least_influential = sorted_factors[-1][0]

if most_influential == 'Sex':
    st.write(" **Gender was the most critical factor** - Women had significantly higher survival rates than men")
elif most_influential == 'Class':
    st.write(" **Class was the most critical factor** - First-class passengers had much better survival rates")
elif most_influential == 'Age':
    st.write(" **Age was the most critical factor** - Children had better survival rates than adults")
else:
    st.write(" **Fare was the most critical factor** - Higher fare passengers had better survival rates")

st.write(f" **{least_influential} had the minimal impact** on survival rates")

st.markdown("---")
st.markdown("*This analysis shows how different factors influenced survival chances in the Titanic disaster.*")
