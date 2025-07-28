# src/Corrected_train/Phase2_8_evaluate_tuned_xgb_top10.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/advanced_features.csv"
MODEL_PATH = "models/corrected_advanced/xgb_tuned_top10.pkl"

# --- 1. LOAD DATA ---
print(f"Loading test data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
df_test = df[df["split"] == "test"].copy()

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
TOP_10_FEATURES = ['brent_oil', 'prev_low', 'gold_price', 'sentiment_x_volatility', 
                   'trading_compound_count_60m', 'bin_slot_number', 
                   'trading_compound_count_180m', 'mean_pre_open', 'volume', 
                   'trading_compound_mean_180m']

# Preprocess the test data
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes
X_test = df_test[TOP_10_FEATURES]
y_test = df_test[TARGET]

# --- 3. LOAD THE TUNED MODEL ---
print(f"Loading the tuned (Top 10) XGBoost model from {MODEL_PATH}...")
tuned_model = joblib.load(MODEL_PATH)

# --- 4. EVALUATE ON TEST DATA ---
def evaluate(X, y, name):
    preds = tuned_model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Tuned XGBoost (Top 10 Features) Performance on Test Data ==")
evaluate(X_test, y_test, "Test Set")