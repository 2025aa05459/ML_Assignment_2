import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Bank Marketing ML App", layout="centered")

st.title("📊 Bank Marketing Classification App")
st.write("Upload a CSV file (bank-additional format) to evaluate model performance.")

# -----------------------------
# Upload File
# -----------------------------

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4)
    }

# -----------------------------
# Main Logic
# -----------------------------

if uploaded_file:

    # Read CSV (semicolon separated)
    df = pd.read_csv(uploaded_file, sep=';')

    # Clean column names
    df.columns = df.columns.str.replace('"', '')
    df.columns = df.columns.str.strip()

    # Check target column
    if "y" not in df.columns:
        st.error("❌ Target column 'y' not found in uploaded file.")
        st.write("Detected columns:", df.columns)
        st.stop()

    # Convert target to numeric
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    y = df["y"]
    X = df.drop("y", axis=1)

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Scale features (important for LR & KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load selected model
    model_filename = model_choice.replace(" ", "_") + ".pkl"
    model_path = f"model/{model_filename}"

    try:
        model = pickle.load(open(model_path, "rb"))
    except:
        st.error(f"❌ Model file not found: {model_filename}")
        st.stop()

    # Make predictions
    if model_choice in ["Logistic Regression", "KNN"]:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    # Evaluate
    metrics = evaluate(y, y_pred, y_prob)

    st.subheader("📈 Evaluation Metrics")
    st.json(metrics)

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

    st.subheader("📋 Classification Report")
    st.text(classification_report(y, y_pred))

    st.success("✅ Model evaluation completed successfully!")
