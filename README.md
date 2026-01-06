# Banking Customer Churn Prediction Using ANN  
## End-to-End Solution: From Business Problem to Deployed Web Application

---

## 1. Business Problem (Banking Domain)

Banks lose significant revenue when customers close their accounts or stop using banking services. Customer churn directly impacts profitability, customer lifetime value, and market competitiveness. Retaining existing customers is substantially cheaper than acquiring new ones.

### Business Challenge
- Identify bank customers who are likely to exit
- Understand churn risk early
- Take proactive retention actions

---

## 2. Business Objective

The objective is to build a predictive system that helps the bank:
- Predict whether a customer will churn
- Estimate the probability of churn
- Support data-driven retention strategies

### Core Business Question
Which banking customers are at high risk of churn, and how confident is that prediction?

---

## 3. Business to Data Science Mapping

| Banking Perspective | Data Science Perspective |
|---------------------|--------------------------|
| Customer exits or stays | Binary classification |
| Churn risk level | Probability score |
| Customer profile | Feature vector |
| Retention decision | Threshold-based output |

---

## 4. Banking Dataset Understanding

The banking dataset contains customer-level attributes such as:

- Credit Score  
- Geography (Country/Region)  
- Gender  
- Age  
- Tenure with the bank  
- Account Balance  
- Number of bank products  
- Credit card ownership  
- Active membership status  
- Estimated salary  

### Target Variable
- `1` → Customer exited (churned)  
- `0` → Customer retained  

---

## 5. Data Preprocessing (Banking Context)

### Why preprocessing is required
- Neural networks require numerical input
- Banking features are on different scales
- Categorical banking attributes must be encoded

### Preprocessing Steps
- Label Encoding for binary categorical features (Gender)
- One-Hot Encoding for multi-class features (Geography)
- Feature Scaling using StandardScaler
- Saving encoders and scaler using pickle for reuse during inference

---

## 6. Model Selection: Artificial Neural Network (ANN)

### Why ANN for banking churn prediction
- Captures complex, non-linear customer behavior
- Learns interactions between financial and behavioral features
- Suitable for large-scale banking datasets
- Provides probabilistic output useful for risk scoring

---

## 7. Model Training (experiment.py)

### Training Process
- Banking customer features fed into ANN
- Hidden layers learn churn-related patterns
- Sigmoid output layer predicts churn probability

### Model Configuration
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Activation Functions: ReLU (hidden layers), Sigmoid (output)

### Saved Artifacts
- Trained ANN model (`model.h5`)
- Feature scaler (`scaler.pkl`)
- Label encoder for Gender
- One-hot encoder for Geography

---

## 8. Model Evaluation (Banking Perspective)

Evaluation focuses on:
- Identifying high-risk churn customers
- Minimizing false negatives (customers predicted as safe but churn)
- Using churn probability instead of only class labels

---

## 9. Single Customer Prediction Pipeline (prediction.py)

Purpose:
- Perform churn prediction for a single banking customer
- Used for validation, testing, and backend services

Flow:
- Accept customer details
- Apply same preprocessing pipeline
- Load trained ANN model
- Output churn probability and decision

---

## 10. Web Application Layer (app.py)

### Why a Banking Web Application
Relationship managers and business teams need an easy-to-use interface for churn analysis without technical knowledge.

### Streamlit Application Features
- User-friendly banking input forms
- Real-time churn probability calculation
- Clear churn / no-churn decision display

---

## 11. Decision-Making Layer (Banking Actions)

Based on churn probability:

| Churn Probability | Banking Action |
|------------------|----------------|
| Low | No immediate action |
| Medium | Personalized communication |
| High | Retention offers, fee waivers, incentives |

---

## 12. Deployment Strategy

- Local deployment using Streamlit
- Cloud deployment planned using Streamlit Cloud
- Same ANN model and preprocessing objects reused

---

## 13. End-to-End Banking System Flow

Customer Banking Data  
→ Data Preprocessing (Encoding + Scaling)  
→ ANN Churn Model  
→ Churn Probability  
→ Risk-Based Decision  
→ Customer Retention Strategy  

---

## 14. Business Impact for Banking

- Reduced customer exit rate
- Improved customer retention
- Higher customer lifetime value
- Proactive relationship management
- Data-driven banking decisions

---

## 15. Project Significance

- Solves a real-world banking problem
- Clear business-to-technical translation
- Covers complete ML lifecycle
- Production-ready and deployable
- Suitable for enterprise banking use cases

---

## Summary

This project transforms a **banking customer churn problem** into an end-to-end ANN-based predictive system, integrating data preprocessing, model training, single-customer inference, and a Streamlit-powered web application to support real-time banking decisions.
