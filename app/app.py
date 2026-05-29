import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="AdaBoost Classifier",
    page_icon="🍷",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #fff0f3; }
h1 { color: #800020; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #800020;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🍷 AdaBoost Classifier - Wine Dataset")
st.write("Classify Wine types using AdaBoost Classifier with tunable hyperparameters")

# Load dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
target_names = data.target_names

df = X.copy()
df["Target"] = y
df["Wine Class"] = df["Target"].map(dict(enumerate(target_names)))

# ---- Sidebar: Hyperparameters ----
st.sidebar.header("⚙️ Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators", 10, 300, 50, step=10)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 2.0, 1.0, step=0.01)
max_depth = st.sidebar.slider("Base Estimator Max Depth", 1, 10, 1)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

base_estimator = DecisionTreeClassifier(max_depth=max_depth)
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=int(random_state)
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("CV Mean Score", f"{cv_scores.mean():.2%}")
col3.metric("Test Samples", len(y_test))
col4.metric("Train Samples", len(y_train))

# Sidebar prediction
st.sidebar.header("🔬 Predict Wine Class")
alcohol = st.sidebar.number_input("Alcohol", 11.0, 15.0, 13.0)
malic_acid = st.sidebar.number_input("Malic Acid", 0.7, 5.8, 2.3)
ash = st.sidebar.number_input("Ash", 1.3, 3.2, 2.3)
flavanoids = st.sidebar.number_input("Flavanoids", 0.3, 5.1, 2.0)
color_intensity = st.sidebar.number_input("Color Intensity", 1.2, 13.0, 5.0)

if st.sidebar.button("Predict Wine Class"):
    sample = X_test.iloc[0].copy()
    sample["alcohol"] = alcohol
    sample["malic_acid"] = malic_acid
    sample["ash"] = ash
    sample["flavanoids"] = flavanoids
    sample["color_intensity"] = color_intensity
    prediction = model.predict([sample.values])
    st.success(f"🍷 Predicted Wine Class: **{target_names[prediction[0]]}**")

# Dataset
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# Confusion Matrix
st.subheader("🔥 Confusion Matrix")
fig1, ax1 = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=target_names, yticklabels=target_names, ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)

# Feature Importance
st.subheader("⭐ Feature Importance")
importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=importance, palette="Reds_r", ax=ax2)
ax2.set_title("Feature Importance")
st.pyplot(fig2)

# Cross Validation Scores
st.subheader("📈 Cross Validation Scores")
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.bar(range(1, 6), cv_scores, color="#800020")
ax3.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean: {cv_scores.mean():.2f}")
ax3.set_xlabel("Fold")
ax3.set_ylabel("Accuracy")
ax3.set_title("5-Fold Cross Validation Scores")
ax3.legend()
st.pyplot(fig3)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap="Reds", fmt=".2f", ax=ax4)
ax4.set_title("Correlation Heatmap")
st.pyplot(fig4)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.markdown("---")
st.markdown("Developed using Streamlit and AdaBoost Classifier")