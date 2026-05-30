import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Stacking Regressor",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0fff4; }
h1 { color: #1b5e20; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #1b5e20;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Stacking Regressor - California Housing")
st.write("Predict house prices using Stacking Ensemble with tunable hyperparameters")

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

df = X.copy()
df["MedHouseVal"] = y

# ---- Sidebar: Hyperparameters ----
st.sidebar.header("⚙️ Hyperparameters")

st.sidebar.subheader("Base Learners")
use_dt = st.sidebar.checkbox("Decision Tree", value=True)
dt_max_depth = st.sidebar.slider("DT Max Depth", 1, 10, 3)

use_knn = st.sidebar.checkbox("KNN", value=True)
knn_neighbors = st.sidebar.slider("KNN Neighbors", 1, 15, 5)

use_rf = st.sidebar.checkbox("Random Forest", value=True)
rf_estimators = st.sidebar.slider("RF Estimators", 10, 200, 50)

use_gbr = st.sidebar.checkbox("Gradient Boosting", value=True)
gbr_estimators = st.sidebar.slider("GBR Estimators", 10, 200, 50)

st.sidebar.subheader("Meta Learner")
meta_alpha = st.sidebar.slider("Meta Learner Alpha (Ridge)", 0.01, 10.0, 1.0)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Build base learners
base_learners = []
if use_dt:
    base_learners.append(("Decision Tree", DecisionTreeRegressor(max_depth=dt_max_depth, random_state=int(random_state))))
if use_knn:
    base_learners.append(("KNN", KNeighborsRegressor(n_neighbors=knn_neighbors)))
if use_rf:
    base_learners.append(("Random Forest", RandomForestRegressor(n_estimators=rf_estimators, random_state=int(random_state))))
if use_gbr:
    base_learners.append(("Gradient Boosting", GradientBoostingRegressor(n_estimators=gbr_estimators, random_state=int(random_state))))

if len(base_learners) == 0:
    st.error("Please select at least one base learner!")
    st.stop()

# Meta learner
meta_learner = Ridge(alpha=meta_alpha)

# Train Stacking model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=cv_folds
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=3, scoring="r2")

st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("R² Score", f"{r2:.4f}")
col4.metric("CV R² Mean", f"{cv_scores.mean():.4f}")

# Individual model comparison
st.subheader("📊 Base Learners vs Stacking Comparison")
comparison_data = []
for name, reg in base_learners:
    reg.fit(X_train, y_train)
    r2_score_val = r2_score(y_test, reg.predict(X_test))
    comparison_data.append({"Model": name, "R² Score": r2_score_val})
comparison_data.append({"Model": "Stacking", "R² Score": r2})
comparison_df = pd.DataFrame(comparison_data).sort_values("R² Score", ascending=False)
st.dataframe(comparison_df)

fig0, ax0 = plt.subplots(figsize=(8, 4))
colors = ["#1b5e20" if m == "Stacking" else "#66bb6a" for m in comparison_df["Model"]]
ax0.bar(comparison_df["Model"], comparison_df["R² Score"], color=colors)
ax0.set_ylabel("R² Score")
ax0.set_title("Model Comparison")
st.pyplot(fig0)

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
ax1.scatter(y_test[:200], y_pred[:200], alpha=0.5, color="#1b5e20")
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")
ax1.set_title("Actual vs Predicted House Values")
st.pyplot(fig1)

# Cross Validation
st.subheader("📈 Cross Validation R² Scores")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(range(1, len(cv_scores)+1), cv_scores, color="#1b5e20")
ax2.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean: {cv_scores.mean():.4f}")
ax2.set_xlabel("Fold")
ax2.set_ylabel("R² Score")
ax2.set_title("Cross Validation R² Scores")
ax2.legend()
st.pyplot(fig2)

# Distribution
st.subheader("📈 House Value Distribution")
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.histplot(y, kde=True, color="#1b5e20", ax=ax3)
ax3.set_title("House Value Distribution")
st.pyplot(fig3)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="Greens", fmt=".2f", ax=ax4)
ax4.set_title("Correlation Heatmap")
st.pyplot(fig4)

# Comparison Table
st.subheader("📋 Actual vs Predicted Table")
comparison = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})
st.dataframe(comparison)

st.markdown("---")
st.markdown("Developed using Streamlit and Stacking Regressor")