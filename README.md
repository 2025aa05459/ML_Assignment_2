# ML Assignment 2 -- Multi-Model Classification & Streamlit Deployment

------------------------------------------------------------------------

## a. Problem Statement

The objective of this project is to implement and compare multiple
supervised machine learning classification algorithms on a real-world
dataset. Each model is evaluated using standardized performance metrics
to analyze predictive strength, robustness, and generalization
capability.

An interactive Streamlit web application was also developed to allow
users to upload datasets, select models, and view evaluation results
dynamically. This project demonstrates a complete end-to-end machine
learning workflow including preprocessing, modeling, evaluation, and
cloud deployment.

------------------------------------------------------------------------

## b. Dataset Description

**Dataset Name:** Adult Census Income Dataset\
**Original Source:** UCI Machine Learning Repository\
**Accessed Via:** Kaggle\
**Number of Instances:** \~48,000\
**Number of Features:** 14\
**Target Variable:** Income Level

-   `<=50K` → Income less than or equal to 50K\
-   `>50K` → Income greater than 50K

### Dataset Overview

The dataset contains demographic and employment-related attributes such
as:

-   Age\
-   Work class\
-   Education\
-   Marital status\
-   Occupation\
-   Capital gain and loss\
-   Hours per week\
-   Native country

The task is a binary classification problem to predict whether an
individual's annual income exceeds 50K.

This dataset satisfies assignment requirements:

✔ More than 500 instances\
✔ More than 12 features\
✔ Suitable for classification modeling

------------------------------------------------------------------------

## c. Models Implemented

The following six classification models were implemented and trained on
the same dataset:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbors (KNN)\
4.  Naive Bayes (Gaussian)\
5.  Random Forest (Ensemble Method)\
6.  XGBoost (implemented using Gradient Boosting due to environment
    constraints)

### Evaluation Metrics Used

-   Accuracy\
-   AUC (Area Under ROC Curve)\
-   Precision\
-   Recall\
-   F1 Score\
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Model Performance Comparison

  ------------------------------------------------------------------------------------
  ML Model     Accuracy    AUC        Precision     Recall     F1 Score     MCC
  ------------ ----------- ---------- ------------- ---------- ------------ ----------
  Logistic     0.830598    0.864158   0.755319      0.472703   0.581491     0.503076
  Regression                                                                

  Decision     0.811039    0.746908   0.620828      0.619174   0.620000     0.494257
  Tree                                                                      

  KNN          0.833250    0.861496   0.684799      0.611851   0.646273     0.539097

  Naive Bayes  0.790817    0.838863   0.667131      0.318908   0.431532     0.355436

  Random       0.863584    0.913552   0.771817      0.641811   0.700836     0.617827
  Forest                                                                    

  XGBoost      0.867230    0.926377   0.808267      0.611851   0.696476     0.623316
  ------------------------------------------------------------------------------------

------------------------------------------------------------------------

## d. Performance Analysis

### Logistic Regression

Provides a strong baseline model with high precision but comparatively
lower recall, indicating conservative predictions.

### Decision Tree

Offers balanced precision and recall but shows lower AUC compared to
ensemble models and may overfit.

### KNN

Performs reasonably well after feature scaling but is sensitive to the
number of neighbors and computationally heavier.

### Naive Bayes

Trains quickly but has lower recall and MCC due to its strong
independence assumption.

### Random Forest

Achieves strong overall performance with improved accuracy and MCC due
to ensemble averaging.

### XGBoost

Delivers the highest AUC and competitive overall metrics, demonstrating
strong predictive capability.

------------------------------------------------------------------------

## Streamlit Web Application

The interactive web application includes:

-   CSV dataset upload\
-   Model selection dropdown\
-   Real-time evaluation metrics display\
-   Confusion matrix\
-   Classification report

------------------------------------------------------------------------

## Deployment

The application was deployed using **Streamlit Community Cloud**,
providing a live and interactive interface for testing and evaluation.

------------------------------------------------------------------------

## Conclusion

This project demonstrates a complete machine learning pipeline from
preprocessing to deployment. Ensemble models (Random Forest and XGBoost)
achieved superior overall performance, particularly in AUC and MCC.

The integration of model comparison and web deployment reflects a
practical real-world machine learning implementation workflow.
