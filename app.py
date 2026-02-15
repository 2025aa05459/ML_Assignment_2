import streamlit as st
import pandas as pd

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

# Import models
from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost

st.set_page_config(page_title="Bank Marketing ML App")

st.title("📊 Bank Marketing Classification App")

st.write("Dataset: bank-additional-full.csv")

# -----------------------------------
# Load Dataset Automatically
# -----------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional/bank-additional-full.csv", sep=';')
    df.columns = df.columns.str.replace('"', '')
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Encode target
df["y"] = df["y"].map({"yes": 1, "no": 0})

y = df["y"]
X = df.drop("y", axis=1)

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Scale for LR & KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# Model Selection
# -----------------------------------

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

# -----------------------------------
# Evaluation Function
# -----------------------------------

def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4)
    }

# -----------------------------------
# Train & Evaluate
# -----------------------------------

if st.button("Train & Evaluate Model"):

    if model_choice == "Logistic Regression":
        model, X_test, y_test = train_logistic_regression(X_scaled, y)

    elif model_choice == "Decision Tree":
        model, X_test, y_test = train_decision_tree(X, y)

    elif model_choice == "KNN":
        model, X_test, y_test = train_knn(X_scaled, y)

    elif model_choice == "Naive Bayes":
        model, X_test, y_test = train_naive_bayes(X, y)

    elif model_choice == "Random Forest":
        model, X_test, y_test = train_random_forest(X, y)

    else:
        model, X_test, y_test = train_xgboost(X, y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob)

    st.subheader("📈 Evaluation Metrics")
    st.json(metrics)

    st.subheader("📉 Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.success("✅ Model evaluation completed successfully!")
