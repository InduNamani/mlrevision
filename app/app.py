import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AdaBoost Regressor",
    page_icon="🏡",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0f7ff; }
h1 { color: #023e8a; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #023e8a;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🏡 AdaBoost Regressor - California Housing")
st.write("Predict house prices using AdaBoost Regressor with tunable hyperparameters")

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

df = X.copy()
df["MedHouseVal"] = y

# ---- Sidebar: Hyperparameters ----
st.sidebar.header("⚙️ Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators", 10, 300, 50, step=10)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, 1.0, step=0.01)
max_depth = st.sidebar.slider("Base Estimator Max Depth", 1, 10, 3)
loss = st.sidebar.selectbox("Loss Function", ["linear", "square", "exponential"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

base_estimator = DecisionTreeRegressor(max_depth=max_depth)
model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    loss=loss,
    random_state=int(random_state)
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("R² Score", f"{r2:.4f}")
col4.metric("CV R² Mean", f"{cv_scores.mean():.4f}")

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
ax1.scatter(y_test[:200], y_pred[:200], alpha=0.5, color="#023e8a")
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
sns.barplot(x="Importance", y="Feature", data=importance, palette="Blues_r", ax=ax2)
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# Cross Validation Scores
st.subheader("📈 Cross Validation R² Scores")
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.bar(range(1, 6), cv_scores, color="#023e8a")
ax3.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean: {cv_scores.mean():.4f}")
ax3.set_xlabel("Fold")
ax3.set_ylabel("R² Score")
ax3.set_title("5-Fold Cross Validation R² Scores")
ax3.legend()
st.pyplot(fig3)

# Distribution
st.subheader("📈 House Value Distribution")
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.histplot(y, kde=True, color="#023e8a", ax=ax4)
ax4.set_title("House Value Distribution")
st.pyplot(fig4)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax5)
ax5.set_title("Correlation Heatmap")
st.pyplot(fig5)

# Comparison Table
st.subheader("📋 Actual vs Predicted Table")
comparison = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})
st.dataframe(comparison)

st.markdown("---")
st.markdown("Developed using Streamlit and AdaBoost Regressor")