import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower Classification Using SVM")

df = pd.read_csv("../data/iris.csv")

encoder = LabelEncoder()

df['species'] = encoder.fit_transform(df['species'])

X = df.drop("species", axis=1)

y = df["species"]

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

st.sidebar.header("Enter Flower Details")

sepal_length = st.sidebar.slider("Sepal Length",4.0,8.0,5.0)

sepal_width = st.sidebar.slider("Sepal Width",2.0,5.0,3.0)

petal_length = st.sidebar.slider("Petal Length",1.0,7.0,2.0)

petal_width = st.sidebar.slider("Petal Width",0.1,3.0,0.2)

if st.sidebar.button("Predict Species"):

    sample_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    sample_data = scaler.transform(sample_data)

    prediction = model.predict(sample_data)

    species_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Species: {species_names[prediction[0]]}"
    )

st.subheader("📊 Model Accuracy")

st.metric(
    label="Accuracy Score",
    value=f"{accuracy:.2f}"
)

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())

st.subheader("📈 Species Count")

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.countplot(
    x='species',
    data=df,
    ax=ax1
)

st.pyplot(fig1)

st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10,7))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    ax=ax2
)

st.pyplot(fig2)

st.subheader("📌 Sepal Length vs Petal Length")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x='sepal_length',
    y='petal_length',
    hue='species',
    data=df,
    ax=ax3
)

st.pyplot(fig3)

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

st.pyplot(fig4)

st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

st.markdown("---")

st.markdown("Developed using Streamlit and SVM")
