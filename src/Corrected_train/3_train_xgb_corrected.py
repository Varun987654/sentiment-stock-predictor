# src/Corrected_train/3_train_xgb_corrected.py

import pandas as pd
import xgboost as xgb
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
df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Convert ticker to a numerical category for XGBoost
df_train['ticker'] = df_train['ticker'].astype('category').cat.codes
df_val['ticker'] = df_val['ticker'].astype('category').cat.codes
df_test['ticker'] = df_test['ticker'].astype('category').cat.codes

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val = df_val[FEATURES]
y_val = df_val[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

print(f"\nTraining XGBoost model on {len(FEATURES)} features...")

# --- 3. TRAIN BASELINE XGBOOST MODEL ---
# Using some standard, untuned parameters as a starting point
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50 # Stop training if validation performance doesn't improve
)

# XGBoost requires an evaluation set to use early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# --- 4. EVALUATE ---
def evaluate(X, y, name):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Baseline XGBoost Performance (Corrected Data) ==")
evaluate(X_val, y_val, "Validation")
evaluate(X_test, y_test, "Test")

# --- 5. SAVE MODEL ---
joblib.dump(model, os.path.join(MODEL_DIR, "xgb_baseline_corrected.pkl"))
print(f"\nSaved corrected XGBoost baseline model to {MODEL_DIR}")