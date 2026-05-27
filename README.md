# E-commerce User Churn Prediction Project

## Overview
This project applies exploratory data analysis (EDA) and machine learning techniques to predict user churn, uncover key churn drivers and generate actionable business recommendations to improve user retention.


## Objective
To identify users with high churn risk so that the business can proactively deploy targeted retention strategies to improve user retention and optimise customer lifetime value.


## Folder Structure
```
Churn Prediction/
│
├── src/
│   └── train.py              # End-to-end ML pipeline
│
├── Dataset/
│   └── E Commerce Dataset.xlsx
│
├── run.sh                   # Execution script
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```
## How to run pipeline

- Install dependencies:
```bash
pip install -r requirements.text
```

- Run full pipeline:
```bash
bash run.sh
```

#### Parameters
The recall threshold for tuning can be modified in `train.py`:
```
RECALL_TARGET = 0.8
```

## Pipeline Flow

- Data loading  
- Preprocessing  
- Train-test split  
- Imputation
- Feature scaling (Logistic Regression only) 
- Model training
- Prediction & Probabilities
- Baseline evaluation
- Threshold tuning (recall ≥ 0.8)  
- Post tuning evaluation
- ROC-AUC comparison
- Final model selection


## EDA Insights & Feature Engineering Decisions
- Early tenure users display the highest churn risk, with tenure showing the strongest negative relation with churn
   - Tenure was treated as a key predictive feature where missing values were imputed using City Tier median as median to preserve segment-level behavioural differences
     
- Complaint behaviour is a strong churn driver with users who lodged complaints exhibiting approximately 3x higher churn compared to those without complaints
   - Feature was retained as a key signal in the model without transformation

- Cash payment has the highest churn rate of 25% indicating potential payment friction compared to other payment methods
   - Payment modes were grouped into 3 broader categories during preprocessing to reduce category fragmentation and improve model stability


## Feature Processing Summary

| Feature Group | Examples | Processing |
|---------------|----------|------------|
| Missing values | Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder | Median imputation, scaling for Logistic Regression |
| Engineered features | PaymentMode | Grouped into 3 categories (Card, Digital & Cash) for stability |
| Numerical features | NumberOfDeviceRegistered, NumberOfAddress, CashbackAmount | Scaling for Logistic Regression |
| Categorical features | PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat, MaritalStatus | One-hot encoding |
| Binary features | Gender | Label encoding (Female=0, Male=1) |
| Identifier | CustomerID | Dropped from model |

## Final Model

Random Forest selected based on best precision-recall trade-off under recall constraint.
