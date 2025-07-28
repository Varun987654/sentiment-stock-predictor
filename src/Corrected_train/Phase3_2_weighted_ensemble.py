# src/Corrected_train/Phase3_2_weighted_ensemble.py

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

# --- 4. CREATE AND EVALUATE WEIGHTED ENSEMBLES ---
print("\n== Evaluating Weighted Ensembles (Test Data) ==")

def evaluate(y_true, y_preds, name):
    mse = mean_squared_error(y_true, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_preds)
    print(f"{name:<25} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

# Test different weights
weights_to_test = [
    (0.5, 0.5), # Simple Average (our previous result)
    (0.7, 0.3), # 70% LGBM, 30% XGB
    (0.8, 0.2), # 80% LGBM, 20% XGB
    (0.9, 0.1)  # 90% LGBM, 10% XGB
]

for lgbm_w, xgb_w in weights_to_test:
    preds_ensemble = (lgbm_w * preds_lgbm) + (xgb_w * preds_xgb)
    evaluate(y_test, preds_ensemble, f"Ensemble ({int(lgbm_w*100)}% LGBM, {int(xgb_w*100)}% XGB)")