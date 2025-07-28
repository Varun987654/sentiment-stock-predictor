# src/Corrected_train/Phase4_3_professional_backtest.py

import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
ADVANCED_DATA_CSV = "data/06_Features_corrected/advanced_features.csv"
LGBM_MODEL_PATH = "models/corrected_advanced/lgbm_tuned_advanced.pkl"
XGB_MODEL_PATH = "models/corrected_advanced/xgb_tuned_regularized_advanced.pkl"
OUTPUT_DIR = "results/professional_backtest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimal weights we discovered from our optimization script
LGBM_WEIGHT = 0.8
XGB_WEIGHT = 0.2
# A realistic transaction cost of 0.05% per trade
TRANSACTION_COST = 0.0005 

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

# Keep original ticker names for analysis
df_test_processed = df_test.copy()
df_test_processed['ticker'] = pd.Categorical(df_test_processed['ticker']).codes
X_test = df_test_processed[FEATURES]


# --- 3. MAKE PREDICTIONS ---
print("Making final ensemble predictions...")
preds_lgbm = lgbm_model.predict(X_test)
preds_xgb = xgb_model.predict(X_test)
final_prediction = (LGBM_WEIGHT * preds_lgbm) + (XGB_WEIGHT * preds_xgb)
df_test['prediction'] = final_prediction

# --- 4. RUN THE PROFESSIONAL BACKTEST ---
print("\nRunning professional backtest with transaction costs...")
# The P&L is the return from our position minus the transaction cost for the trade
df_test['pnl'] = (np.sign(df_test['prediction']) * df_test[TARGET]) - TRANSACTION_COST

# --- 5. OVERALL PERFORMANCE ---
df_test['date'] = pd.to_datetime(df_test['bin_end_time']).dt.date
daily_pnl = df_test.groupby('date')['pnl'].sum()
daily_sharpe = 0
if daily_pnl.std() != 0 and daily_pnl.std() is not np.nan:
    daily_sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
cumulative_pnl = df_test['pnl'].sum()

print("\n== Final Overall Backtest Performance ==")
print(f"   Sharpe Ratio (Daily, After Costs): {daily_sharpe:.4f}")
print(f"   Cumulative P&L (After Costs):      {cumulative_pnl:.4f}")

# --- 6. PERFORMANCE BY TICKER ---
print("\n== Performance Breakdown by Ticker ==")
for ticker in sorted(df_test['ticker'].unique()):
    ticker_df = df_test[df_test['ticker'] == ticker]
    daily_pnl_ticker = ticker_df.groupby('date')['pnl'].sum()
    
    ticker_sharpe = 0
    if daily_pnl_ticker.std() != 0 and daily_pnl_ticker.std() is not np.nan:
        ticker_sharpe = (daily_pnl_ticker.mean() / daily_pnl_ticker.std()) * np.sqrt(252)
        
    ticker_pnl = ticker_df['pnl'].sum()
    print(f"\n   --- Ticker: {ticker} ---")
    print(f"      Sharpe Ratio: {ticker_sharpe:.4f}")
    print(f"      Cumulative P&L: {ticker_pnl:.4f}")

# --- 7. SAVE RESULTS ---
output_path = os.path.join(OUTPUT_DIR, "professional_backtest_results.csv")
df_test[['ticker', 'bin_end_time', TARGET, 'prediction', 'pnl']].to_csv(output_path, index=False)
print(f"\nSaved final professional backtest results to {output_path}")