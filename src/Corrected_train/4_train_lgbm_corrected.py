# src/Corrected_train/4_train_lgbm_corrected.py

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/merged_features_split_corrected_final.csv"
MODEL_DIR = "models/corrected"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
# Use .copy() to avoid SettingWithCopyWarning
df_train = df[df["split"] == "train"].copy()
df_val = df[df["split"] == "val"].copy()
df_test = df[df["split"] == "test"].copy()

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Convert ticker to a numerical category for LightGBM
df_train['ticker'] = pd.Categorical(df_train['ticker']).codes
df_val['ticker'] = pd.Categorical(df_val['ticker']).codes
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val = df_val[FEATURES]
y_val = df_val[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

print(f"\nTraining LightGBM model on {len(FEATURES)} features...")

# --- 3. TRAIN BASELINE LIGHTGBM MODEL ---
# Using some standard, untuned parameters as a starting point
model = lgb.LGBMRegressor(
    objective='regression_l1', # MAE is often more robust to outliers
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# LightGBM requires a callback for early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# --- 4. EVALUATE ---
def evaluate(X, y, name):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Baseline LightGBM Performance (Corrected Data) ==")
print(f"   (Best iteration: {model.best_iteration_})")
evaluate(X_val, y_val, "Validation")
evaluate(X_test, y_test, "Test")

# --- 5. SAVE MODEL ---
joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_baseline_corrected.pkl"))
print(f"\nSaved corrected LightGBM baseline model to {MODEL_DIR}")