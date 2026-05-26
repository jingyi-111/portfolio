# Churn Prediction Project

## Overview
ML pipeline to predict customer churn using Logistic Regression, Random Forest, and XGBoost.

## How to run
```bash
bash run.sh
```

## Pipeline Steps

- Data loading  
- Preprocessing  
- Train-test split  
- Imputation  
- Model training  
- Threshold tuning (recall ≥ 0.8)  
- Evaluation  

## Final Model

Random Forest selected based on best precision-recall trade-off under recall constraint.
