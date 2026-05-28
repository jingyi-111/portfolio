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
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# =========================
# 1. CONFIGURATION
# =========================

DATA_PATH = 'Dataset/E Commerce Dataset.xlsx'
SHEET_NAME = 'E Comm'

TEST_SIZE = 0.3
RANDOM_STATE = 42
RECALL_TARGET = 0.81

TARGET = "Churn"


# =========================
# 2. LOAD DATA
# =========================

def load_data(): 

	return pd.read_excel(DATA_PATH, sheet_name= SHEET_NAME)


# =========================
# 3. PREPROCESSING
# =========================

def preprocess(df):

	# Drop identifier
	df_model = df.drop(columns='CustomerID')
	
	# Standardise categories
	df_model['PreferredLoginDevice'] = df_model['PreferredLoginDevice'].replace('Phone','Mobile Phone')
	df_model['PreferedOrderCat'] = df_model['PreferedOrderCat'].replace('Mobile','Mobile Phone')
	
	# Grouped categories
	df_model['PreferredPaymentMode'] = df_model['PreferredPaymentMode'].replace({
    	'CC':'Card',
    	'COD':'Cash',
    	'Cash on Delivery':'Cash',
    	'Credit Card':'Card',
    	'Debit Card':'Card',
    	'E wallet':'Digital',
    	'UPI':'Digital'
	})

	# One-hot encoding
	df_model = pd.get_dummies(df_model, columns=['PreferredLoginDevice','PreferredPaymentMode','PreferedOrderCat','MaritalStatus'],drop_first=True)

	# Label encoding
	gender_map = {'Female':0,'Male':1}
	df_model['Gender'] = df_model['Gender'].map(gender_map)	

	return df_model


# =========================
# 4. TRAIN TEST SPLIT
# =========================

def split_data(df):

	X = df.drop(columns=TARGET,axis=1)
	y = df[TARGET]

	return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# =========================
# 5. IMPUTATION & SCALING
# =========================

def impute_and_scale(df, X_train, X_test):

	null_values = df.isnull().mean()
	null_values_cols = null_values[null_values > 0].index

	# Tenure imputation by city tier
	city_median = X_train.groupby('CityTier')['Tenure'].median()

	X_train = X_train.copy()
	X_test = X_test.copy()

	X_train['Tenure'] = X_train['Tenure'].fillna(X_train['CityTier'].map(city_median))

	X_test['Tenure'] = X_test['Tenure'].fillna(X_test['CityTier'].map(city_median))

	null_values_cols = null_values_cols.drop('Tenure')

	# Imputation for remaining missing values
	num_imputer = SimpleImputer(strategy='median')
	X_train[null_values_cols] = num_imputer.fit_transform(X_train[null_values_cols])
	X_test[null_values_cols] = num_imputer.transform(X_test[null_values_cols])


	# Scaling for Logistic Regression only
	sc = StandardScaler()

	X_train_scaled = X_train.copy()
	X_test_scaled = X_test.copy()

	sc_col = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp',
       	'NumberOfDeviceRegistered', 'NumberOfAddress',
       	'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
       	'DaySinceLastOrder', 'CashbackAmount']

	X_train_scaled[sc_col] = sc.fit_transform(X_train_scaled[sc_col])
	X_test_scaled[sc_col] = sc.transform(X_test_scaled[sc_col])

	return X_train, X_test, X_train_scaled, X_test_scaled


# =========================
# 6. TRAIN MODELS
# =========================

def train_models(X_train, X_train_scaled, y_train):
	
	models = {}

	models['logistic_regression'] = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)

	models['random_forest'] = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')

	models['xgboost'] = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)

	models['logistic_regression'].fit(X_train_scaled,y_train)

	models['random_forest'].fit(X_train,y_train)

	models['xgboost'].fit(X_train,y_train)
	
	return models


# =========================
# 7. THRESHOLD TUNING FUNCTION
# =========================

def tune_threshold(y_true, y_proba, recall_target):

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    idx = np.where(recall[:-1] >= recall_target)[0]

    best_idx = idx[np.argmax(precision[idx])]

    best_threshold = thresholds[best_idx]

    y_pred = (y_proba >= best_threshold).astype(int)

    return best_threshold, y_pred


# =========================
# 8. EVALUATION
# =========================

def evaluate(models, X_test, X_test_scaled, y_test):
	
	summary = {}

	for name, model in models.items():

		if name == 'logistic_regression':
			y_pred = model.predict(X_test_scaled)
			y_proba = model.predict_proba(X_test_scaled)[:,1]
		
		else:	
			y_pred = model.predict(X_test)
			y_proba = model.predict_proba(X_test)[:,1]	

		best_threshold, y_pred_tune = tune_threshold(y_test, y_proba, RECALL_TARGET)
		auc = roc_auc_score(y_test, y_proba)

		print(f"\n============ {name.upper()} ============")
		print("\n========== BASELINE EVALUATION ==========")
		print("Threshold: default")
		print(classification_report(y_test, y_pred))

		print(f"\nROC-AUC: {auc:.3f}")

		print("\n========== POST TUNING EVALUATION ==========")
		print(f"Threshold: {best_threshold:.3f}")
		print(classification_report(y_test, y_pred_tune))
		print("============================================")

		report = classification_report(y_test, y_pred_tune, output_dict=True)
		summary[name] = report['1']['f1-score']
	
	best_model = max(summary, key=summary.get)

	print("\n==============================")
	print("FINAL MODEL SELECTION:", best_model.upper())
	print("==============================")

	return best_model


# =========================
# 9. MAIN PIPELINE
# =========================

def main():

	df = load_data() 

	df_model = preprocess(df)

	X_train, X_test, y_train, y_test = split_data(df_model)

	X_train, X_test, X_train_scaled, X_test_scaled = impute_and_scale(df_model, X_train, X_test)

	models = train_models(X_train, X_train_scaled, y_train)

	evaluate(models, X_test, X_test_scaled, y_test)


# =========================
# 10. ENTRY POINT
# =========================

if __name__ == '__main__':
	main() 


