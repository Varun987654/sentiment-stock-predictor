# src/Corrected_train/Phase3_1_final_ensemble.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIG ---
ADVANCED_DATA_CSV = "data/06_Features_corrected/advanced_features.csv"
LGBM_MODEL_PATH = "models/corrected_advanced/lgbm_tuned_advanced.pkl"
XGB_MODEL_PATH = "models/corrected_advanced/xgb_tuned_regularized_advanced.pkl"

# --- 1. LOAD DATA & MODELS ---
print("Loading data and both best models...")
df_adv = pd.read_csv(ADVANCED_DATA_CSV)
lgbm_model = joblib.load(LGBM_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

# Use only the test split
df_test = df_adv[df_adv["split"] == "test"].copy()

# --- 2. PREPARE FEATURE SET ---
print("Preparing feature set...")
TARGET = 'target_return'
FEATURES = [col for col in df_adv.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Preprocess data
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# --- 3. MAKE PREDICTIONS ---
print("Making predictions with both models...")
preds_lgbm = lgbm_model.predict(X_test)
preds_xgb = xgb_model.predict(X_test)

# --- 4. CREATE ENSEMBLE PREDICTION (SIMPLE AVERAGE) ---
print("Creating ensemble prediction...")
preds_ensemble = (preds_lgbm + preds_xgb) / 2.0

# --- 5. EVALUATE ENSEMBLE ---
def evaluate(y_true, y_preds, name):
    mse = mean_squared_error(y_true, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Final Ensemble Performance (Test Data) ==")
evaluate(y_test, preds_ensemble, "Ensemble")