import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Decision Tree Regressor",
    page_icon="🌲",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f5f0eb; }
h1 { color: #7f4f24; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #7f4f24;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🌲 Decision Tree Regressor - California Housing")
st.write("Predict house prices using Decision Tree Regression")

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

df = X.copy()
df["MedHouseVal"] = y

# Sidebar
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 15, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 5)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 2)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("R² Score", f"{r2:.4f}")

# Sidebar prediction
st.sidebar.header("🏠 Predict House Value")
MedInc = st.sidebar.number_input("Median Income", 0.5, 15.0, 3.5)
HouseAge = st.sidebar.number_input("House Age", 1.0, 52.0, 20.0)
AveRooms = st.sidebar.number_input("Avg Rooms", 1.0, 20.0, 5.0)
AveBedrms = st.sidebar.number_input("Avg Bedrooms", 1.0, 5.0, 1.0)
Population = st.sidebar.number_input("Population", 100.0, 5000.0, 1000.0)
AveOccup = st.sidebar.number_input("Avg Occupancy", 1.0, 10.0, 3.0)
Latitude = st.sidebar.number_input("Latitude", 32.0, 42.0, 35.0)
Longitude = st.sidebar.number_input("Longitude", -125.0, -114.0, -119.0)

if st.sidebar.button("Predict House Value"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)
    st.success(f"🏡 Predicted House Value: **${prediction[0]*100000:,.2f}**")

# Dataset
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# Actual vs Predicted
st.subheader("📌 Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(y_test[:200], y_pred[:200], alpha=0.5, color="#7f4f24")
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted House Values")
st.pyplot(fig1)

# Feature Importance
st.subheader("⭐ Feature Importance")
importance = pd.DataFrame({
    "Feature": housing.feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(x="Importance", y="Feature", data=importance, palette="YlOrBr_r", ax=ax2)
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# Distribution
st.subheader("📈 House Value Distribution")
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.histplot(y, kde=True, color="#7f4f24", ax=ax3)
ax3.set_title("House Value Distribution")
st.pyplot(fig3)

# Comparison Table
st.subheader("📋 Actual vs Predicted Table")
comparison = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})
st.dataframe(comparison)

st.markdown("---")
st.markdown("Developed using Streamlit and Decision Tree Regressor")