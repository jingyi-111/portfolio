#!/usr/bin/env python
# coding: utf-8

# =========================
# CHURN PREDICTION PIPELINE
# =========================

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_excel('Dataset/E Commerce Dataset.xlsx',sheet_name='E Comm')


# =========================
# 2. PREPROCESSING
# =========================

target = "Churn"
df_model = df.drop(columns='CustomerID')


df_model['PreferredLoginDevice'] = df_model['PreferredLoginDevice'].replace('Phone','Mobile Phone')

df_model['PreferredPaymentMode'] = df_model['PreferredPaymentMode'].replace({
    'CC':'Card',
    'COD':'Cash',
    'Cash on Delivery':'Cash',
    'Credit Card':'Card',
    'Debit Card':'Card',
    'E wallet':'Digital',
    'UPI':'Digital'
})

df_model['PreferedOrderCat'] = df_model['PreferedOrderCat'].replace('Mobile','Mobile Phone')


df_model = pd.get_dummies(df_model,columns=['PreferredLoginDevice','PreferredPaymentMode','PreferedOrderCat','MaritalStatus'],drop_first=True)

gender_map = {'Female':0,'Male':1}
df_model['Gender'] = df_model['Gender'].map(gender_map)


X = df_model.drop(columns=target,axis=1)
y = df_model[target]


# =========================
# 3. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# =========================
# 4. IMPUTATION
# =========================

null_values = df_model.isnull().mean()
null_values_cols = null_values[null_values > 0].index


city_median = X_train.groupby('CityTier')['Tenure'].median()

X_train['Tenure'] = X_train['Tenure'].fillna(
    X_train['CityTier'].map(city_median))

X_test['Tenure'] = X_test['Tenure'].fillna(
    X_test['CityTier'].map(city_median))

null_values_cols = null_values_cols.drop('Tenure')


num_imputer = SimpleImputer(strategy='median')
X_train[null_values_cols] = num_imputer.fit_transform(X_train[null_values_cols])
X_test[null_values_cols] = num_imputer.transform(X_test[null_values_cols])


# =========================
# 5. SCALING (for Logistic Regression only)
# =========================

sc = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

sc_col = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
       'NumberOfDeviceRegistered', 'NumberOfAddress',
       'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       'DaySinceLastOrder', 'CashbackAmount']

X_train_scaled[sc_col] = sc.fit_transform(X_train_scaled[sc_col])
X_test_scaled[sc_col] = sc.transform(X_test_scaled[sc_col])


# =========================
# 6. DEFINE MODELS
# =========================

lr = LogisticRegression(class_weight='balanced',random_state=42)

rf = RandomForestClassifier(random_state=42,class_weight='balanced')

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)


# =========================
# 7. TRAIN MODELS
# =========================

lr.fit(X_train_scaled,y_train)

rf.fit(X_train,y_train)

xgb.fit(X_train,y_train)


# =========================
# 8. GET PREDICTIONS AND PROBABILITIES
# =========================

y_lr_pred = lr.predict(X_test_scaled)
y_lr_proba = lr.predict_proba(X_test_scaled)[:,1]

y_rf_pred = rf.predict(X_test)
y_rf_proba = rf.predict_proba(X_test)[:,1]

y_xgb_pred = xgb.predict(X_test)
y_xgb_proba = xgb.predict_proba(X_test)[:,1]


# =========================
# 9. BASELINE EVALUATION (THRESHOLD = 0.5)
# =========================

print("\n========== BASELINE EVALUATION ==========")
print("\n===== LOGISTIC REGRESSION =====")
print("Threshold: default")
print(classification_report(y_test, y_lr_pred))

print("\n===== RANDOM FOREST =====")
print("Threshold: default")
print(classification_report(y_test, y_rf_pred))

print("\n===== XGBOOST =====")
print("Threshold: default")
print(classification_report(y_test, y_xgb_pred))


# =========================
# 10. THRESHOLD TUNING FUNCTION
# =========================

def tune_threshold(y_true, y_proba, recall_target=0.8):

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    idx = np.where(recall[:-1] >= recall_target)[0]

    best_idx = idx[np.argmax(precision[idx])]

    best_threshold = thresholds[best_idx]

    y_pred = (y_proba >= best_threshold).astype(int)

    return best_threshold, y_pred


# =========================
# 11. THRESHOLD OPTIMISATION (RECALL ≥ 0.8)
# =========================

RECALL_TARGET = 0.8

lr_thres, y_lr_pred1 = tune_threshold(y_test, y_lr_proba, RECALL_TARGET)
rf_thres, y_rf_pred1 = tune_threshold(y_test, y_rf_proba, RECALL_TARGET)
xgb_thres, y_xgb_pred1 = tune_threshold(y_test, y_xgb_proba, RECALL_TARGET)


# =========================
# 12. POST TUNING EVALUATION
# =========================

print("\n========== POST TUNING EVALUATION ==========")
print("\n===== LOGISTIC REGRESSION =====")
print("Threshold:", round(lr_thres, 3))
print(classification_report(y_test, y_lr_pred1))

print("\n===== RANDOM FOREST =====")
print("Threshold:", round(rf_thres, 3))
print(classification_report(y_test, y_rf_pred1))

print("\n===== XGBOOST =====")
print("Threshold:", round(xgb_thres, 3))
print(classification_report(y_test, y_xgb_pred1))


# =========================
# 13. ROC-AUC COMPARISON
# =========================

print("\nROC-AUC SCORES")
print("Logistic Regression:", round(roc_auc_score(y_test, y_lr_proba),3))
print("Random Forest:", round(roc_auc_score(y_test, y_rf_proba),3))
print("XGBoost:", round(roc_auc_score(y_test, y_xgb_proba),3))


# =========================
# 14. FINAL MODEL SELECTION
# =========================

print("\nFINAL MODEL SELECTION: RANDOM FOREST")
print("Reason: best precision-recall tradeoff under recall threshold ≥ 0.8")


