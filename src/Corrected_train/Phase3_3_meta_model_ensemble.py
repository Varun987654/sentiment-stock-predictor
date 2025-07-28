# src/Corrected_train/Phase3_3_meta_model_ensemble.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIG ---
ADVANCED_DATA_CSV = "data/06_Features_corrected/advanced_features.csv"
LGBM_MODEL_PATH = "models/corrected_advanced/lgbm_tuned_advanced.pkl"
XGB_MODEL_PATH = "models/corrected_advanced/xgb_tuned_regularized_advanced.pkl"
META_MODEL_DIR = "models/corrected_advanced"
os.makedirs(META_MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA & MODELS ---
print("Loading data and both best models...")
df_adv = pd.read_csv(ADVANCED_DATA_CSV)
lgbm_model = joblib.load(LGBM_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)

# Separate splits for training the meta-model (val) and final evaluation (test)
df_val = df_adv[df_adv["split"] == "val"].copy()
df_test = df_adv[df_adv["split"] == "test"].copy()

# --- 2. PREPARE FEATURE SETS ---
print("Preparing feature sets...")
TARGET = 'target_return'
FEATURES = [col for col in df_adv.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Preprocess data for both validation and test sets
for df_split in [df_val, df_test]:
    df_split['ticker'] = pd.Categorical(df_split['ticker']).codes

X_val = df_val[FEATURES]
y_val = df_val[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# --- 3. MAKE PREDICTIONS ON VALIDATION SET (to train the meta-model) ---
print("Making validation set predictions to create training data for meta-model...")
val_preds_lgbm = lgbm_model.predict(X_val)
val_preds_xgb = xgb_model.predict(X_val)

# This is the training data for our meta-model
X_train_meta = pd.DataFrame({
    'lgbm_pred': val_preds_lgbm,
    'xgb_pred': val_preds_xgb
})

# --- 4. TRAIN THE META-MODEL ---
print("Training the meta-model to find the optimal weights...")
meta_model = LinearRegression()
meta_model.fit(X_train_meta, y_val)

# --- 5. SHOW THE OPTIMAL WEIGHTS ---
lgbm_weight = meta_model.coef_[0]
xgb_weight = meta_model.coef_[1]
print(f"\nOptimal weights found by meta-model:")
print(f"  - LightGBM Weight: {lgbm_weight:.4f}")
print(f"  - XGBoost Weight:  {xgb_weight:.4f}")

# --- 6. MAKE PREDICTIONS ON TEST SET ---
print("\nMaking test set predictions with base models...")
test_preds_lgbm = lgbm_model.predict(X_test)
test_preds_xgb = xgb_model.predict(X_test)

# Create the input data for the meta-model
X_test_meta = pd.DataFrame({
    'lgbm_pred': test_preds_lgbm,
    'xgb_pred': test_preds_xgb
})

# Make final predictions with the trained meta-model
final_preds = meta_model.predict(X_test_meta)

# --- 7. EVALUATE THE META-MODEL ENSEMBLE ---
def evaluate(y_true, y_preds, name):
    mse = mean_squared_error(y_true, y_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Final Meta-Model Ensemble Performance (Test Data) ==")
evaluate(y_test, final_preds, "Meta-Ensemble")

joblib.dump(meta_model, os.path.join(META_MODEL_DIR, "meta_model_ensemble.pkl"))
print(f"\nSaved meta-model to {META_MODEL_DIR}")