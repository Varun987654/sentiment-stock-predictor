# src/Corrected_train/Phase2_5_evaluate_tuned_xgb_advanced.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/advanced_features.csv"
MODEL_PATH = "models/corrected_advanced/xgb_tuned_advanced.pkl"

# --- 1. LOAD DATA ---
print(f"Loading test data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
df_test = df[df["split"] == "test"].copy()

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Preprocess the test data in the same way
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# --- 3. LOAD THE TUNED MODEL ---
print(f"Loading the tuned XGBoost model from {MODEL_PATH}...")
tuned_model = joblib.load(MODEL_PATH)

# --- 4. EVALUATE ON TEST DATA ---
def evaluate(X, y, name):
    preds = tuned_model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Tuned XGBoost Performance on Test Data ==")
evaluate(X_test, y_test, "Test Set")