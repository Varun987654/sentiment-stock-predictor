# src/backtest2.py

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# ─── 1) Load & preprocess ─────────────────────────────
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_Features', 'merged_features_split.csv')
df = pd.read_csv(csv_path, parse_dates=['bin_end_time'])

# [… same momentum, VIX/IRX, encoding, rolling features as before …]

# f) Features & splits
low_var_feats = [ … ]  # same list
exclude = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date'] + low_var_feats
X_all = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])
y_all = df['bin_return']
splits = df['split']

# ─── 2) Split & scale ───────────────────────────────────
with open(os.path.join(project_root, 'models', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

X_trainval = scaler.transform(X_all[splits != 'test'])
y_trainval = y_all[splits != 'test'].values
X_test     = scaler.transform(X_all[splits == 'test'])
y_test     = y_all[splits == 'test'].values

# Keep the dates for grouping
test_dates = df.loc[splits=='test', 'bin_end_time'].dt.date

# ─── 3) Load base models ─────────────────────────────────
with open(os.path.join(project_root, 'models', 'lr.pkl'), 'rb') as f:
    lr = pickle.load(f)
with open(os.path.join(project_root, 'models', 'xgb.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)
with open(os.path.join(project_root, 'models', 'lgbm.pkl'), 'rb') as f:
    lgbm = pickle.load(f)

pred_lr  = lr.predict(X_test)
pred_xgb = xgb_model.predict(X_test)
pred_lgb = lgbm.predict(X_test)

# ─── 4) Simple‑average ensemble ───────────────────────────
pred_avg = (pred_lr + pred_xgb + pred_lgb) / 3

# ─── 5) Build P&L signals ───────────────────────────────
bt = pd.DataFrame({
    'date':     test_dates,
    'true':     y_test,
    'avg_pred': pred_avg
})
bt['signal'] = np.sign(bt['avg_pred'])
bt['pnl']    = bt['signal'] * bt['true']

# ─── 6) Compute daily P&L & Sharpe (correctly) ───────────
#  a) average P&L per bar across tickers to get one series per day
daily_bar_pnl = bt.groupby('date')['pnl'].mean()
#  b) daily Sharpe (mean/std of daily_pnl × sqrt(252))
daily_sharpe  = daily_bar_pnl.mean() / daily_bar_pnl.std() * np.sqrt(252)
#  c) cumulative P&L (sum of all bar P&Ls, or equivalently sum of daily_bar_pnl)
cum_pnl       = bt['pnl'].sum()

# ─── 7) Print results ────────────────────────────────────
print("\n=== Simple‑Average Ensemble ===")
print("RMSE:           ", np.sqrt(mean_squared_error(y_test, pred_avg)))
print("R²:             ", r2_score(y_test, pred_avg))
print("Daily Sharpe:   ", daily_sharpe)
print("Cumulative P&L: ", cum_pnl)
