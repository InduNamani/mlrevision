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
from sklearn.linear_model import Ridge
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

# Load and sample dataset for faster performance
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseVal")
    # Sample 3000 rows for faster loading
    idx = np.random.RandomState(42).choice(len(X), 3000, replace=False)
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    return X, y

X, y = load_data()
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
rf_estimators = st.sidebar.slider("RF Estimators", 10, 100, 50)

use_gbr = st.sidebar.checkbox("Gradient Boosting", value=True)
gbr_estimators = st.sidebar.slider("GBR Estimators", 10, 100, 50)

st.sidebar.subheader("Meta Learner")
meta_alpha = st.sidebar.slider("Meta Learner Alpha (Ridge)", 0.01, 10.0, 1.0)
cv_folds = st.sidebar.slider("CV Folds", 2, 5, 3)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Cache model training
@st.cache_resource
def train_model(use_dt, dt_max_depth, use_knn, knn_neighbors,
                use_rf, rf_estimators, use_gbr, gbr_estimators,
                meta_alpha, cv_folds, test_size, random_state):

    base_learners = []
    if use_dt:
        base_learners.append(("Decision Tree", DecisionTreeRegressor(max_depth=dt_max_depth, random_state=random_state)))
    if use_knn:
        base_learners.append(("KNN", KNeighborsRegressor(n_neighbors=knn_neighbors)))
    if use_rf:
        base_learners.append(("Random Forest", RandomForestRegressor(n_estimators=rf_estimators, random_state=random_state)))
    if use_gbr:
        base_learners.append(("Gradient Boosting", GradientBoostingRegressor(n_estimators=gbr_estimators, random_state=random_state)))

    meta_learner = Ridge(alpha=meta_alpha)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=cv_folds
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, base_learners

if not (use_dt or use_knn or use_rf or use_gbr):
    st.error("Please select at least one base learner!")
    st.stop()

with st.spinner("Training Stacking model... please wait ⏳"):
    model, X_train, X_test, y_train, y_test, base_learners = train_model(
        use_dt, dt_max_depth, use_knn, knn_neighbors,
        use_rf, rf_estimators, use_gbr, gbr_estimators,
        meta_alpha, cv_folds, test_size, int(random_state)
    )

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

# Individual model comparison
st.subheader("📊 Base Learners vs Stacking Comparison")
comparison_data = []
for name, reg in base_learners:
    reg.fit(X_train, y_train)
    r2_val = r2_score(y_test, reg.predict(X_test))
    comparison_data.append({"Model": name, "R² Score": round(r2_val, 4)})
comparison_data.append({"Model": "Stacking", "R² Score": round(r2, 4)})
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

# Distribution
st.subheader("📈 House Value Distribution")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(y, kde=True, color="#1b5e20", ax=ax2)
ax2.set_title("House Value Distribution")
st.pyplot(fig2)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="Greens", fmt=".2f", ax=ax3)
ax3.set_title("Correlation Heatmap")
st.pyplot(fig3)

# Comparison Table
st.subheader("📋 Actual vs Predicted Table")
comparison = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})
st.dataframe(comparison)

st.markdown("---")
st.markdown("Developed using Streamlit and Stacking Regressor")