# Week 4 - Credit Card Fraud Detector imbalance handling, SMOTE

## 📌 Overview

This module focuses on handling the **class imbalance problem** in the Credit Card Fraud Detection dataset.

Credit card fraud datasets are highly imbalanced because fraudulent transactions are much fewer than genuine transactions. Training a machine learning model directly on such data can cause the model to favor the majority class and fail to detect fraudulent transactions effectively.

In this module, the dataset is cleaned and preprocessed, a baseline Logistic Regression model is trained, and **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to balance the training data. The model is then retrained and evaluated to analyze the impact of SMOTE on fraud detection.

---

## 🎯 Objectives

- Load and preprocess the Credit Card Fraud Detection dataset.
- Remove duplicate and missing records.
- Save the cleaned dataset for further processing.
- Split the dataset into training and testing sets.
- Train a baseline Logistic Regression model.
- Evaluate the model on the original imbalanced dataset.
- Apply SMOTE to handle class imbalance.
- Train the model using the balanced dataset.
- Compare the performance before and after SMOTE.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

**Dataset:** Credit Card Fraud Detection  
**Source:** Kaggle

The dataset contains:

- 284,807 transactions
- 31 features
- 284,315 genuine transactions
- 492 fraudulent transactions

The target column is:

- `0` → Genuine Transaction
- `1` → Fraudulent Transaction

### Dataset Link

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

### Machine Learning Algorithm

- Logistic Regression

### Class Imbalance Technique

- SMOTE

---

## 🔄 Methodology

```text
Credit Card Dataset
        ↓
Data Cleaning
        ↓
Remove Duplicates & Missing Values
        ↓
Save Cleaned Dataset
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Baseline Evaluation
        ↓
Apply SMOTE
        ↓
Balanced Training Data
        ↓
Logistic Regression
        ↓
Final Evaluation
        ↓
Before vs After SMOTE Comparison
