# E-commerce User Churn Prediction Project

## Overview
This project applies exploratory data analysis (EDA) and machine learning techniques to predict user churn, uncover key churn drivers and generate actionable business recommendations to improve user retention.


## Objective
To identify users with high churn risk so that the business can proactively deploy targeted retention strategies to improve user retention, optimise customer lifetime value and control unnecessasry retention cost.


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
RECALL_TARGET = 0.81
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
- Threshold tuning (recall ≥ 0.81)  
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


## Model Selection

- Logistic Regression
   - Served as a baseline model for performance benchmark due to its simplicity and interpretability, assuming a linear relationship between features and churn probability

- Random Forest
   - Selected as a non-linear tree-based model that performs well on imbalanced dataset
   - Ability to capture complex feature interactions without requiring feature scaling

- XGBoost 
   - Used as an advanced boosting model to evaluate potential performance improvement beyond Random Forest and to assess the upper bound of model performance


## Model Evaluation

Model performance was evaluated using the following metrics:

- Precision: Measures how many predicted churn users were actually churners  
- Recall: Measures how many actual churn users were correctly identified  
- F1-score: Harmonic mean of precision and recall, balancing both objectives  
- ROC-AUC: Measures the model’s ability to distinguish between churn and non-churn users

Given the business objective of identifying churn users while controlling retention costs, recall was prioritised during threshold tuning. A recall threshold of ≥ 0.81 was set based on baseline performance observed from Logistic Regression, ensuring that subsequent models meet or exceed this minimum recall level.

### Results Summary

#### Pre-Tuning Performance Summary

| Model | Precision | Recall | F1-score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.44 | 0.81 | 0.57 | 0.88 |
| Random Forest | 0.95 | 0.79 | 0.86 | 0.99 |
| XGBoost | 0.91 | 0.82 | 0.87 | 0.99 |

#### Post-Tuning Performance Summary
| Model | Precision | Recall | F1-score | Threshold |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.45 | 0.81 | 0.58 | 0.50 |
| Random Forest | 0.94 | 0.82 | 0.88 | 0.47 |
| XGBoost | 0.92 | 0.81 | 0.86 | 0.51 |

<img width="589" height="451" alt="Screenshot 2026-05-28 at 1 14 32 PM" src="https://github.com/user-attachments/assets/a7142ae0-eaab-414a-ba77-cc2cecc4b750" />

#### Key Insights

- All models achieved strong ROC-AUC scores (0.88–0.99), indicating good class separability between churn and non-churn users
- Tree-based models (Random Forest and XGBoost) consistently outperformed Logistic Regression in overall predictive performance
- Random Forest demonstrated a more balanced precision-recall trade-off as observed from the Precision-Recall curve while XGBoost showed stronger performance prior to tuning
- Threshold tuning was applied on all models using the same recall constraint to optimise churn detection performance and to ensure a fair comparison


### Final Model Selection

Random Forest was selected as the final model as it achieved the best balance between precision (0.94) and recall (0.82) after threshold tuning, making it the most suitable model for churn identification under the business constraint.



