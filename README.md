# Telco Customer Churn Prediction: Sampling Techniques
Machine learning analysis that will be investigating how different sampling
strategies would affect customer churn prediction under class imbalance.

## Overview
Customer churn is a major problem in many industries, including the telecommunications industry.
This project will investigate whether resampling techniques can improve
the ability of machine learning models to identify customers who churn.

The study compares:

- Decision Tree
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

across multiple sampling strategies:

- No Sampling
- SMOTE
- SMOTE-Tomek
- SMOTE-ENN
- ADASYN
- Cluster-SMOTE
- NearMiss
- Instance Hardness Threshold
- VAE-SMOTE
- GAN-based oversampling

The models are evaluated primarily using F1-score and ROC-AUC,
with particular attention to the minority churn class.

---

## Problem

Telecom churn datasets are typically imbalanced, meaning that
customers who leave represent a minority of the dataset.

In the dataset used for this project, approximately 26% of customers
churned while 74% did not.

This imbalance can cause models to favour the majority class and
perform poorly when identifying churners.

---

## Objectives

1. Analyse the class imbalance in telecom customer churn data.
2. Implement multiple sampling techniques.
3. Compare four machine learning classifiers.
4. Evaluate the effect of sampling on minority-class prediction.
5. Compare models using F1-score and ROC-AUC.

---

## Dataset

The dataset contains more than 7,000 telecom customer records.

Features include:

- Customer demographics
- Tenure
- Contract type
- Payment method
- Internet service
- Monthly charges
- Total charges
- Service subscriptions

Target:

`Churn`

The project uses an 80/20 train-test split with stratification.

---

## Methodology

The project follows the CRISP-DM framework:

Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Sampling
        ↓
Model Development
        ↓
Evaluation
        ↓
Analysis

### Data preprocessing

- Removed customer identifiers
- Converted `TotalCharges` to numeric
- Handled missing values
- Encoded categorical variables
- Scaled numerical variables
- Used stratified train/test splitting

### Sampling techniques

SMOTE  
SMOTE-Tomek  
SMOTE-ENN  
ADASYN  
Cluster-SMOTE  
NearMiss  
Instance Hardness Threshold

### Models

Decision Tree  
Logistic Regression  
Random Forest  
Support Vector Machine

---

## Results

### F1-score

| Sampling Method | Decision Tree | Logistic Regression | Random Forest | SVM |
|---|---:|---:|---:|---:|
| No Sampling | 0.481 | 0.612 | 0.529 | 0.562 |
| SMOTE | 0.525 | 0.617 | 0.545 | 0.618 |
| SMOTE-Tomek | 0.529 | 0.616 | 0.561 | 0.618 |
| SMOTE-ENN | 0.585 | 0.586 | 0.609 | 0.590 |
| ADASYN | 0.507 | 0.612 | 0.552 | 0.609 |
| Cluster-SMOTE | 0.517 | 0.644 | 0.556 | 0.589 |
| NearMiss | 0.405 | 0.556 | 0.434 | 0.481 |
| IHT | 0.563 | 0.569 | 0.567 | 0.567 |

### Key observations

- Sampling affected the classifiers differently.
- SMOTE-ENN substantially improved the Decision Tree and Random Forest
  F1-scores compared with their unsampled baselines.
- Cluster-SMOTE produced the highest Logistic Regression F1-score
  in the experiments.
- SMOTE produced a strong SVM F1-score.
- NearMiss reduced F1-score across all four models.
- Sampling therefore cannot be treated as a one-size-fits-all
  preprocessing step.

---

## Technologies

Python  
Pandas  
NumPy  
Scikit-learn  
imbalanced-learn  
Matplotlib  
Seaborn  
Jupyter Notebook

---
