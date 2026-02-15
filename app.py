import streamlit as st
import pandas as pd
import pickle
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

st.title("Bank Marketing Classification App")

uploaded_file = st.file_uploader("Upload Test CSV (Must contain column 'y')", type=["csv"])

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

def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "y" not in df.columns:
        st.error("Uploaded file must contain target column named 'y'")
        st.stop()

    y = df["y"]
    X = df.drop("y", axis=1)

    # Load model
    model_path = f"model/{model_choice.replace(' ', '_')}.pkl"
    model = pickle.load(open(model_path, "rb"))

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = evaluate(y, y_pred, y_prob)

    st.subheader("Evaluation Metrics")
    st.write(metrics)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
