import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="KNN Classifier", layout="centered")

st.title("🌸 KNN Classifier – Iris Dataset")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y
df["Flower Name"] = df["Target"].apply(lambda x: target_names[x])

# Sidebar controls
st.sidebar.header("⚙️ Model Settings")
k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train, y_train)

# Feature inputs
st.sidebar.header("🌱 Input Features")
inputs = []
for i, feature in enumerate(feature_names):
    value = st.sidebar.slider(
        feature,
        float(X[:, i].min()),
        float(X[:, i].max()),
        float(X[:, i].mean())
    )
    inputs.append(value)

input_data = np.array([inputs])

# Prediction
prediction = model.predict(input_data)[0]
accuracy = accuracy_score(y_test, model.predict(x_test))

# Output
st.success(f"Predicted Flower: **{target_names[prediction]}**")
st.write(f"🎯 Model Accuracy: **{accuracy * 100:.2f}%**")

# View dataset
with st.expander("📂 View Iris Dataset"):
    st.dataframe(df)
