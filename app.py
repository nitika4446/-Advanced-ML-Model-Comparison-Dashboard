import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="ML Model Comparison Dashboard", layout="wide")

st.title("🤖 Advanced ML Model Comparison Dashboard")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Mall_Customers.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Data Cleaning + Target Creation
    # -------------------------------
    df = df.drop(columns=["CustomerID"])

    # Create target (High spender vs Low spender)
    df["Target"] = df["Spending Score (1-100)"].apply(lambda x: 1 if x > 50 else 0)

    st.write("### 📊 Dataset Preview", df.head())

    # -------------------------------
    # Features & Target
    # -------------------------------
    target = "Target"

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical to numeric
    X = pd.get_dummies(X)

    # -------------------------------
    # Train Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Scaling
    # -------------------------------
    scaler_option = st.selectbox(
        "⚙️ Select Scaling Method",
        ["None", "StandardScaler", "Normalizer"]
    )

    scaler = None

    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif scaler_option == "Normalizer":
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # -------------------------------
    # Models
    # -------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    st.write("## 📊 Model Comparison")

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results.append((name, acc))

        trained_models[name] = model

        st.write(f"### {name}")
        st.write(f"Accuracy: {acc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.text(classification_report(y_test, preds))

    # -------------------------------
    # Results Table
    # -------------------------------
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.write("### 📈 Model Comparison Table", results_df)

    # -------------------------------
    # Best Model
    # -------------------------------
    best_model = results_df.loc[results_df["Accuracy"].idxmax()]
    st.success(f"🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.write("## 🔮 Make Prediction")

    input_data = {}

    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=0.0)

    input_df = pd.DataFrame([input_data])

    # Apply scaling to input
    if scaler is not None:
        input_df = scaler.transform(input_df)

    selected_model_name = st.selectbox("Choose Model", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]

    if st.button("Predict"):
        prediction = selected_model.predict(input_df)
        result = "High Spender 💰" if prediction[0] == 1 else "Low Spender 🛍️"
        st.success(f"Prediction: {result}")

    # -------------------------------
    # Download Results
    # -------------------------------
    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Results",
        csv,
        "model_results.csv",
        "text/csv"
    )