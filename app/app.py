import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="KNN Regressor", layout="centered")

st.title("📈 KNN Regressor")

# Generate regression dataset
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2

# Convert to DataFrame
df = pd.DataFrame({
    "X": X.squeeze(),
    "Y": y
})

# Sidebar controls
st.sidebar.header("⚙️ Model Settings")
k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = KNeighborsRegressor(n_neighbors=k)
model.fit(x_train, y_train)

# Input
st.sidebar.header("📌 Input")
x_input = st.sidebar.slider(
    "Enter X value",
    float(X.min()),
    float(X.max()),
    float(X.mean())
)

input_data = np.array([[x_input]])

# Prediction
prediction = model.predict(input_data)[0]
r2 = r2_score(y_test, model.predict(x_test))

# Output
st.success(f"Predicted Output: **{prediction:.2f}**")
st.write(f"📊 R² Score: **{r2:.2f}**")

# View dataset
with st.expander("📂 View Regression Dataset"):
    st.dataframe(df)
