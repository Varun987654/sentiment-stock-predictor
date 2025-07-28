# src/ensemble_train.py

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 1. Load & preprocess exactly as in train_lgbm_es.py
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_Features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# Inject momentum features
df['raw_return'] = (df['close'] - df['open']) / df['open']
df['mom_5']      = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(5))
df = df.dropna(subset=['raw_return', 'mom_5'])

# Merge VIX & IRX (as before)
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

# Prepare features & target
low_var_feats = [
    'pre_open_compound_mean','lagged_pre_open_compound_mean',
    'after_close_compound_std','lagged_trading_compound_mean',
    'trading_compound_mean','lagged_after_close_compound_mean',
    'after_close_compound_mean','pre_open_compound_std'
]
exclude = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date'] + low_var_feats
X = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])
y = df['bin_return']
splits = df['split']

# Split and scale
X_trainval = X[splits != 'test']
y_trainval = y[splits != 'test']
X_test     = X[splits == 'test']
y_test     = y[splits == 'test']

scaler = StandardScaler().fit(X_trainval)

# Persist the scaler
with open(os.path.join(project_root, 'models', 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)


X_trainval = scaler.transform(X_trainval)
X_test     = scaler.transform(X_test)

# Ensure models/ directory exists
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)

# 2. Train Linear Regression
lr = LinearRegression()
lr.fit(X_trainval, y_trainval)
with open(os.path.join(project_root, 'models', 'lr.pkl'), 'wb') as f:
    pickle.dump(lr, f)

# 3. Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_trainval, y_trainval)
with open(os.path.join(project_root, 'models', 'xgb.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)

# 4. Train LightGBM
#    Use best_rounds from your CV (e.g. 91)
lgbm = LGBMRegressor(
    n_estimators=91,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_trainval, y_trainval)
with open(os.path.join(project_root, 'models', 'lgbm.pkl'), 'wb') as f:
    pickle.dump(lgbm, f)

print("✅ Base models trained and saved to models/")  
