import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0f4f8; }
h1 { color: #2d6a4f; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #2d6a4f;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

st.title("🌳 Decision Tree Classifier - Iris Dataset")
st.write("Classify Iris flower species using Decision Tree")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
target_names = iris.target_names

df = X.copy()
df["Target"] = y
df["Species"] = df["Target"].map(dict(enumerate(target_names)))

# Sidebar
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
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
st.sidebar.header("🌸 Predict Species")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.number_input("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.sidebar.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"🌸 Predicted Species: **{target_names[prediction[0]].capitalize()}**")

# Dataset
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

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

# Decision Tree Plot
st.subheader("🌳 Decision Tree Visualization")
fig2, ax2 = plt.subplots(figsize=(16, 8))
plot_tree(model, feature_names=iris.feature_names,
          class_names=target_names, filled=True, ax=ax2)
ax2.set_title("Decision Tree")
st.pyplot(fig2)

# Feature Importance
st.subheader("⭐ Feature Importance")
importance = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.barplot(x="Importance", y="Feature", data=importance, palette="Greens_r", ax=ax3)
ax3.set_title("Feature Importance")
st.pyplot(fig3)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.markdown("---")
st.markdown("Developed using Streamlit and Decision Tree Classifier")