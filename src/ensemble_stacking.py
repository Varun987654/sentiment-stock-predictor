# src/ensemble_stacking.py

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 1) Load & preprocess data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_Features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# Inject momentum features
df['raw_return'] = (df['close'] - df['open']) / df['open']
df['mom_5']      = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(5))
df = df.dropna(subset=['raw_return', 'mom_5'])

# Merge VIX & IRX
df['date_dt'] = pd.to_datetime(df['date'])
for ticker, col in [("^VIX", "vix_close"), ("^IRX", "irx_close")]:
    hist = yf.Ticker(ticker).history(
        start="2025-01-01", end="2025-07-10", interval="1d"
    )[['Close']].rename(columns={'Close': col})
    if hist is None or hist.empty:
        continue
    hist.index.name = 'Date'
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
    df = df.merge(hist, left_on='date_dt', right_on='Date', how='left').ffill()

# Encode categoricals
df['ticker_id']  = LabelEncoder().fit_transform(df['ticker'])
df['session_id'] = LabelEncoder().fit_transform(df['session'])

# Time features
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day']        = dates.dt.day
df['month']      = dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)

# Rolling/window features
df['roll_ret_3']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_ret_5']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).mean())
df['vol_3']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).std())
df['vol_5']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).std())
df['roll_sent_3']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_sent_5']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(5).mean())
df['int_ret3_sent'] = df['roll_ret_3'] * df['trading_compound_mean']

df = df.dropna(subset=[
    'roll_ret_3','roll_ret_5','vol_3','vol_5',
    'roll_sent_3','roll_sent_5','int_ret3_sent'
])

# Prepare features & splits
low_var_feats = [
    'pre_open_compound_mean','lagged_pre_open_compound_mean',
    'after_close_compound_std','lagged_trading_compound_mean',
    'trading_compound_mean','lagged_after_close_compound_mean',
    'after_close_compound_mean','pre_open_compound_std'
]
exclude = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date'] + low_var_feats
X_all = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])
y_all = df['bin_return']
splits = df['split']

# Split and scale
with open(os.path.join(project_root, 'models', 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# **Convert to NumPy arrays so integer indexing works**
X_trainval = scaler.transform(X_all[splits != 'test'])
y_trainval = y_all[splits != 'test'].values
X_test     = scaler.transform(X_all[splits == 'test'])
y_test     = y_all[splits == 'test'].values

# 2) Load base models
with open(os.path.join(project_root, 'models', 'lr.pkl'), 'rb') as f:
    lr = pickle.load(f)
with open(os.path.join(project_root, 'models', 'xgb.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)
with open(os.path.join(project_root, 'models', 'lgbm.pkl'), 'rb') as f:
    lgbm = pickle.load(f)

models = [lr, xgb_model, lgbm]

# 3) Generate out‑of‑fold predictions
tscv = TimeSeriesSplit(n_splits=5)
P = np.zeros((len(y_trainval), len(models)))

for i, m in enumerate(models):
    for train_idx, val_idx in tscv.split(X_trainval):
        # now works because both arrays are NumPy
        m.fit(X_trainval[train_idx], y_trainval[train_idx])
        P[val_idx, i] = m.predict(X_trainval[val_idx])

# 4) Train Ridge meta‑learner
meta = Ridge(alpha=1.0)
meta.fit(P, y_trainval)

# 5) Stacked test predictions
P_test = np.column_stack([m.predict(X_test) for m in models])
pred_stack = meta.predict(P_test)

# 6) Evaluate
rmse_stack = np.sqrt(mean_squared_error(y_test, pred_stack))
r2_stack   = r2_score(y_test, pred_stack)
print(f"Stacking Ensemble → RMSE: {rmse_stack:.5f}, R²: {r2_stack:.5f}")
