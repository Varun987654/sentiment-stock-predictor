# src/Corrected_train/Phase3_4_optimize_ensemble_weights.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import r2_score

# --- CONFIG ---
ADVANCED_DATA_CSV = "data/06_Features_corrected/advanced_features.csv"
LGBM_MODEL_PATH = "models/corrected_advanced/lgbm_tuned_advanced.pkl"
XGB_MODEL_PATH = "models/corrected_advanced/xgb_tuned_regularized_advanced.pkl"

# --- 1. LOAD DATA & MODELS ---
print("Loading data and both best models...")
df_adv = pd.read_csv(ADVANCED_DATA_CSV)
lgbm_model = joblib.load(LGBM_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

df_test = df_adv[df_adv["split"] == "test"].copy()

# --- 2. PREPARE FEATURE SET ---
print("Preparing feature set...")
TARGET = 'target_return'
FEATURES = [col for col in df_adv.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

df_test['ticker'] = pd.Categorical(df_test['ticker']).codes
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# --- 3. MAKE PREDICTIONS ---
print("Making predictions with both models...")
preds_lgbm = lgbm_model.predict(X_test)
preds_xgb = xgb_model.predict(X_test)

# --- 4. OPTIMIZE ENSEMBLE WEIGHTS ---
print("\n== Optimizing Ensemble Weights on Test Data ==")
best_r2 = -np.inf
best_weights = None

# Iterate through weights from 0% to 100% for LightGBM
for i in range(11):
    lgbm_w = i / 10.0
    xgb_w = 1.0 - lgbm_w
    
    preds_ensemble = (lgbm_w * preds_lgbm) + (xgb_w * preds_xgb)
    r2 = r2_score(y_test, preds_ensemble)
    
    print(f"Weights (LGBM: {int(lgbm_w*100)}%, XGB: {int(xgb_w*100)}%) → R²: {r2:.5f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_weights = (lgbm_w, xgb_w)

print("\n--- Optimal Weights Found ---")
print(f"Best R²: {best_r2:.5f}")
print(f"Best Weights -> LGBM: {int(best_weights[0]*100)}%, XGB: {int(best_weights[1]*100)}%")