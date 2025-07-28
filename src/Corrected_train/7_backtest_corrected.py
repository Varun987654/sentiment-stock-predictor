# src/Corrected_train/7_backtest_corrected.py

import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/merged_features_split_corrected_final.csv"
MODEL_PATH = "models/corrected/lgbm_tuned_corrected.pkl"
OUTPUT_DIR = "results/corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. LOAD DATA AND MODEL ---
print("Loading data and the final tuned model...")
df = pd.read_csv(INPUT_CSV, parse_dates=['bin_end_time'])
df_test = df[df["split"] == "test"].copy()

model = joblib.load(MODEL_PATH)

# --- 2. MAKE PREDICTIONS ON THE TEST SET ---
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Preprocess the test data in the same way as before
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes
X_test = df_test[FEATURES]

df_test['prediction'] = model.predict(X_test)

# --- 3. RUN THE BACKTEST ---
print("\nRunning backtest...")
# The actual return is the target_return column
# We take the sign of the prediction to simulate a simple long/short strategy
df_test['pnl'] = np.sign(df_test['prediction']) * df_test[TARGET]

# Calculate daily P&L for Sharpe Ratio calculation
df_test['date'] = pd.to_datetime(df_test['bin_end_time']).dt.date
daily_pnl = df_test.groupby('date')['pnl'].sum()

# Calculate final metrics
# Note: Sharpe Ratio is annualized (multiplied by sqrt of 252 business days)
daily_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
cumulative_pnl = df_test['pnl'].sum()

print("\n== Final Backtest Performance ==")
print(f"   Sharpe Ratio (Daily): {daily_sharpe:.4f}")
print(f"   Cumulative P&L:       {cumulative_pnl:.4f}")

# --- 4. SAVE BACKTEST RESULTS ---
backtest_results = df_test[['ticker', 'bin_end_time', 'target_return', 'prediction', 'pnl']]
output_path = os.path.join(OUTPUT_DIR, "backtest_results_corrected.csv")
backtest_results.to_csv(output_path, index=False)
print(f"\nSaved detailed backtest results to {output_path}")