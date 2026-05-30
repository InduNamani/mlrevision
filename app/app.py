import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Stacking Classifier",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0f4ff; }
h1 { color: #1a237e; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #1a237e;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Stacking Classifier - Breast Cancer Dataset")
st.write("Classify Breast Cancer using Stacking Ensemble with tunable hyperparameters")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
target_names = data.target_names

df = X.copy()
df["Target"] = y
df["Diagnosis"] = df["Target"].map({0: "Malignant", 1: "Benign"})

# ---- Sidebar: Hyperparameters ----
st.sidebar.header("⚙️ Hyperparameters")

st.sidebar.subheader("Base Learners")
use_dt = st.sidebar.checkbox("Decision Tree", value=True)
dt_max_depth = st.sidebar.slider("DT Max Depth", 1, 10, 3)

use_knn = st.sidebar.checkbox("KNN", value=True)
knn_neighbors = st.sidebar.slider("KNN Neighbors", 1, 15, 5)

use_svm = st.sidebar.checkbox("SVM", value=True)
svm_c = st.sidebar.slider("SVM C", 0.1, 10.0, 1.0)

use_rf = st.sidebar.checkbox("Random Forest", value=True)
rf_estimators = st.sidebar.slider("RF Estimators", 10, 200, 50)

st.sidebar.subheader("Meta Learner")
meta_c = st.sidebar.slider("Meta Learner C (LogReg)", 0.01, 10.0, 1.0)
cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Build base learners
base_learners = []
if use_dt:
    base_learners.append(("Decision Tree", DecisionTreeClassifier(max_depth=dt_max_depth, random_state=int(random_state))))
if use_knn:
    base_learners.append(("KNN", KNeighborsClassifier(n_neighbors=knn_neighbors)))
if use_svm:
    base_learners.append(("SVM", SVC(C=svm_c, probability=True, random_state=int(random_state))))
if use_rf:
    base_learners.append(("Random Forest", RandomForestClassifier(n_estimators=rf_estimators, random_state=int(random_state))))

if len(base_learners) == 0:
    st.error("Please select at least one base learner!")
    st.stop()

# Meta learner
meta_learner = LogisticRegression(C=meta_c, max_iter=1000, random_state=int(random_state))

# Train Stacking model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=cv_folds
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stacking Accuracy", f"{accuracy:.2%}")
col2.metric("CV Mean Score", f"{cv_scores.mean():.2%}")
col3.metric("Test Samples", len(y_test))
col4.metric("Train Samples", len(y_train))

# Individual model comparison
st.subheader("📊 Base Learners vs Stacking Comparison")
comparison_data = []
for name, clf in base_learners:
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    comparison_data.append({"Model": name, "Accuracy": acc})
comparison_data.append({"Model": "Stacking", "Accuracy": accuracy})
comparison_df = pd.DataFrame(comparison_data).sort_values("Accuracy", ascending=False)
st.dataframe(comparison_df)

fig0, ax0 = plt.subplots(figsize=(8, 4))
colors = ["#1a237e" if m == "Stacking" else "#7986cb" for m in comparison_df["Model"]]
ax0.bar(comparison_df["Model"], comparison_df["Accuracy"], color=colors)
ax0.set_ylabel("Accuracy")
ax0.set_title("Model Comparison")
ax0.set_ylim(0.8, 1.0)
st.pyplot(fig0)

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)

# Cross Validation Scores
st.subheader("📈 Cross Validation Scores")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(range(1, 6), cv_scores, color="#1a237e")
ax2.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean: {cv_scores.mean():.2f}")
ax2.set_xlabel("Fold")
ax2.set_ylabel("Accuracy")
ax2.set_title("5-Fold Cross Validation Scores")
ax2.legend()
st.pyplot(fig2)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Diagnosis Distribution
st.subheader("📈 Diagnosis Distribution")
fig3, ax3 = plt.subplots(figsize=(6, 4))
df["Diagnosis"].value_counts().plot(kind="bar", color=["#7986cb", "#1a237e"], ax=ax3)
ax3.set_title("Diagnosis Distribution")
ax3.set_xlabel("Diagnosis")
ax3.set_ylabel("Count")
st.pyplot(fig3)

st.markdown("---")
st.markdown("Developed using Streamlit and Stacking Classifier")