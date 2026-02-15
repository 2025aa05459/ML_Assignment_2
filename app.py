import streamlit as st
import pandas as pd
import numpy as np

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

# Import models
from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.gradient_boosting import train_gradient_boosting

st.title("Bank Marketing Classification App")

st.write("Upload test CSV file (with same structure as training dataset)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

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

    y = df["y"]
    X = df.drop("y", axis=1)

    if model_choice == "Logistic Regression":
        model, X_test, y_test = train_logistic_regression(X, y)

    elif model_choice == "Decision Tree":
        model, X_test, y_test = train_decision_tree(X, y)

    elif model_choice == "KNN":
        model, X_test, y_test = train_knn(X, y)

    elif model_choice == "Naive Bayes":
        model, X_test, y_test = train_naive_bayes(X, y)

    elif model_choice == "Random Forest":
        model, X_test, y_test = train_random_forest(X, y)

    else:
        model, X_test, y_test = train_gradient_boosting(X, y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob)

    st.subheader("Evaluation Metrics")
    st.write(metrics)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
