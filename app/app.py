import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌲",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0f4f8; }
h1 { color: #1b4332; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #1b4332;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🌲 Random Forest Classifier - Breast Cancer Dataset")
st.write("Classify Breast Cancer as Malignant or Benign using Random Forest")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
target_names = data.target_names

df = X.copy()
df["Target"] = y
df["Diagnosis"] = df["Target"].map({0: "Malignant", 1: "Benign"})

# Sidebar
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Test Samples", len(y_test))
col3.metric("Train Samples", len(y_train))

# Sidebar prediction
st.sidebar.header("🔬 Predict Diagnosis")
mean_radius = st.sidebar.number_input("Mean Radius", 6.0, 30.0, 14.0)
mean_texture = st.sidebar.number_input("Mean Texture", 9.0, 40.0, 19.0)
mean_perimeter = st.sidebar.number_input("Mean Perimeter", 40.0, 200.0, 92.0)
mean_area = st.sidebar.number_input("Mean Area", 140.0, 2500.0, 655.0)
mean_smoothness = st.sidebar.number_input("Mean Smoothness", 0.05, 0.20, 0.10)

if st.sidebar.button("Predict Diagnosis"):
    sample = X_test.iloc[0].copy()
    sample["mean radius"] = mean_radius
    sample["mean texture"] = mean_texture
    sample["mean perimeter"] = mean_perimeter
    sample["mean area"] = mean_area
    sample["mean smoothness"] = mean_smoothness
    prediction = model.predict([sample.values])
    result = "Benign ✅" if prediction[0] == 1 else "Malignant ⚠️"
    st.success(f"🔬 Predicted Diagnosis: **{result}**")

# Dataset
st.subheader("📁 Dataset Preview")
st.dataframe(df[["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "Diagnosis"]].head())

# Confusion Matrix
st.subheader("🔥 Confusion Matrix")
fig1, ax1 = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=target_names, yticklabels=target_names, ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)

# Feature Importance
st.subheader("⭐ Feature Importance (Top 10)")
importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=importance, palette="Greens_r", ax=ax2)
ax2.set_title("Top 10 Feature Importance")
st.pyplot(fig2)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Distribution
st.subheader("📈 Diagnosis Distribution")
fig3, ax3 = plt.subplots(figsize=(6, 4))
df["Diagnosis"].value_counts().plot(kind="bar", color=["#74c69d", "#1b4332"], ax=ax3)
ax3.set_title("Diagnosis Distribution")
ax3.set_xlabel("Diagnosis")
ax3.set_ylabel("Count")
st.pyplot(fig3)

st.markdown("---")
st.markdown("Developed using Streamlit and Random Forest Classifier")