import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #7a0177;
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #7a0177;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}

</style>
""", unsafe_allow_html=True)


st.title("🩺 Breast Cancer Detection Using SVM")

st.write("Predict whether the cancer is Malignant or Benign")


df = pd.read_csv(r"C:\Users\indun\Downloads\archive (15)\data.csv")


df = df.drop(['id', 'Unnamed: 32'], axis=1)


encoder = LabelEncoder()

df['diagnosis'] = encoder.fit_transform(df['diagnosis'])


X = df.drop("diagnosis", axis=1)

y = df["diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


model = SVC(kernel='linear')

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)


st.sidebar.header("Enter Tumor Details")


radius_mean = st.sidebar.slider(
    "Radius Mean",
    5.0,
    30.0,
    15.0
)

texture_mean = st.sidebar.slider(
    "Texture Mean",
    5.0,
    40.0,
    15.0
)

perimeter_mean = st.sidebar.slider(
    "Perimeter Mean",
    40.0,
    200.0,
    90.0
)

area_mean = st.sidebar.slider(
    "Area Mean",
    100.0,
    2500.0,
    700.0
)

smoothness_mean = st.sidebar.slider(
    "Smoothness Mean",
    0.05,
    0.20,
    0.10
)

compactness_mean = st.sidebar.slider(
    "Compactness Mean",
    0.01,
    0.40,
    0.10
)

concavity_mean = st.sidebar.slider(
    "Concavity Mean",
    0.0,
    0.50,
    0.10
)

concave_points_mean = st.sidebar.slider(
    "Concave Points Mean",
    0.0,
    0.30,
    0.05
)

symmetry_mean = st.sidebar.slider(
    "Symmetry Mean",
    0.10,
    0.50,
    0.20
)

fractal_dimension_mean = st.sidebar.slider(
    "Fractal Dimension Mean",
    0.04,
    0.15,
    0.06
)


if st.sidebar.button("Predict Cancer Type"):

    sample_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,

        1.095,
        0.9053,
        8.589,
        153.4,
        0.006399,
        0.04904,
        0.05373,
        0.01587,
        0.03003,
        0.006193,

        25.38,
        17.33,
        184.6,
        2019.0,
        0.1622,
        0.6656,
        0.7119,
        0.2654,
        0.4601,
        0.1189
    ]])

    sample_data = scaler.transform(sample_data)

    prediction = model.predict(sample_data)

    if prediction[0] == 1:
        st.error("⚠️ Malignant Cancer Detected")

    else:
        st.success("✅ Benign Cancer Detected")


st.subheader("📊 Model Accuracy")

st.metric(
    label="Accuracy Score",
    value=f"{accuracy:.2f}"
)


st.subheader("📁 Dataset Preview")

st.dataframe(df.head())


st.subheader("📈 Diagnosis Count")

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.countplot(
    x='diagnosis',
    data=df,
    ax=ax1
)

ax1.set_title("Diagnosis Count")

st.pyplot(fig1)


st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(14,10))

correlation = df.corr()

sns.heatmap(
    correlation,
    cmap='coolwarm',
    ax=ax2
)

ax2.set_title("Correlation Heatmap")

st.pyplot(fig2)


st.subheader("📌 Radius Mean Distribution")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.histplot(
    df['radius_mean'],
    kde=True,
    ax=ax3
)

ax3.set_title("Radius Mean Distribution")

st.pyplot(fig3)


st.subheader("📋 Actual vs Predicted")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(comparison.head(20))


st.subheader("🧾 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig4, ax4 = plt.subplots(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax4
)

ax4.set_title("Confusion Matrix")

ax4.set_xlabel("Predicted")

ax4.set_ylabel("Actual")

st.pyplot(fig4)


st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)


st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

fig5, ax5 = plt.subplots(figsize=(10,8))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance,
    ax=ax5
)

ax5.set_title("Feature Importance")

st.pyplot(fig5)


st.markdown("---")

st.markdown(
    "Developed using Streamlit and Support Vector Machine"
)