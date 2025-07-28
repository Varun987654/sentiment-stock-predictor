# src/Corrected_train/Phase4_2_final_ensemble_backtest.py

import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
ADVANCED_DATA_CSV = "data/06_Features_corrected/advanced_features.csv"
LGBM_MODEL_PATH = "models/corrected_advanced/lgbm_tuned_advanced.pkl"
XGB_MODEL_PATH = "models/corrected_advanced/xgb_tuned_regularized_advanced.pkl"
OUTPUT_DIR = "results/final_backtest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimal weights we discovered
LGBM_WEIGHT = 0.8
XGB_WEIGHT = 0.2

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

# --- 3. MAKE PREDICTIONS ---
print("Making predictions with both models...")
preds_lgbm = lgbm_model.predict(X_test)
preds_xgb = xgb_model.predict(X_test)

# --- 4. CREATE THE OPTIMAL ENSEMBLE PREDICTION ---
print("Creating the optimal weighted ensemble prediction...")
final_prediction = (LGBM_WEIGHT * preds_lgbm) + (XGB_WEIGHT * preds_xgb)
df_test['prediction'] = final_prediction

# --- 5. RUN THE FINAL BACKTEST ---
print("Running final backtest...")
# Strategy: take the sign of the prediction (1 for buy, -1 for sell)
df_test['pnl'] = np.sign(df_test['prediction']) * df_test[TARGET]

# Calculate daily P&L
df_test['date'] = pd.to_datetime(df_test['bin_end_time']).dt.date
daily_pnl = df_test.groupby('date')['pnl'].sum()

# Calculate final metrics (annualized Sharpe Ratio)
daily_sharpe = 0
if daily_pnl.std() != 0:
    daily_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    
cumulative_pnl = df_test['pnl'].sum()

print("\n== Final Project Backtest Performance (Optimal Ensemble) ==")
print(f"   Sharpe Ratio (Daily): {daily_sharpe:.4f}")
print(f"   Cumulative P&L:       {cumulative_pnl:.4f}")

# --- 6. SAVE RESULTS ---
backtest_results = df_test[['ticker', 'bin_end_time', TARGET, 'prediction', 'pnl']]
output_path = os.path.join(OUTPUT_DIR, "final_ensemble_backtest_results.csv")
backtest_results.to_csv(output_path, index=False)
print(f"\nSaved final backtest results to {output_path}")