# src/backtest_simple_scaled.py
import yfinance as yf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and preprocess data (inline)
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')

df = pd.read_csv(csv_path)
df['bin_return'] = df['bin_return'].clip(-0.05, 0.05)

import yfinance as yf

# Fetch VIX
vix_ticker = yf.Ticker("^VIX")
vix_hist   = vix_ticker.history(start="2025-01-01", end="2025-07-10", interval="1d")
vix = vix_hist[['Close']].rename(columns={'Close':'vix_close'})
vix.index.name = 'Date'
vix = vix.reset_index()

vix['Date'] = vix['Date'].dt.tz_localize(None)

# Merge VIX
df['date_dt'] = pd.to_datetime(df['date'])
df = df.merge(vix, left_on='date_dt', right_on='Date', how='left')
df['vix_close'] = df['vix_close'].fillna(method='ffill')

irx = yf.Ticker("^IRX") \
    .history(start="2025-01-01", end="2025-07-10", interval="1d")[['Close']] \
    .rename(columns={'Close':'irx_close'})
irx.index.name = 'Date'
irx = irx.reset_index()
# drop timezone so it matches df['date_dt']
irx['Date'] = irx['Date'].dt.tz_localize(None)

# Merge IRX into main DataFrame
df = df.merge(irx, left_on='date_dt', right_on='Date', how='left').ffill()

# encode
df['ticker_id']  = LabelEncoder().fit_transform(df['ticker'])
df['session_id'] = LabelEncoder().fit_transform(df['session'])

# time features
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day']        = dates.dt.day
df['month']      = dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)

# rolling features
df['roll_ret_3']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_ret_5']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).mean())
df['vol_3']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).std())
df['vol_5']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).std())
df['roll_sent_3']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_sent_5']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(5).mean())
df['int_ret3_sent'] = df['roll_ret_3'] * df['trading_compound_mean']

# drop NaNs
df = df.dropna(subset=[
    'roll_ret_3','roll_ret_5','vol_3','vol_5',
    'roll_sent_3','roll_sent_5','int_ret3_sent'
])

# prepare dates for folds
unique_dates = sorted(df['date'].unique())
n_folds = 5
fold_size = len(unique_dates) // n_folds

all_rmse = []
all_r2   = []

for i in range(n_folds):
    test_dates = unique_dates[i*fold_size:(i+1)*fold_size]
    train_df = df[~df['date'].isin(test_dates)]
    test_df  = df[df['date'].isin(test_dates)]

    # build X, y
    drop_cols = ['ticker','session','bin_end_time','date','bin_return','split']
    X_train = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
    y_train = train_df['bin_return']
    X_test  = test_df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
    y_test  = test_df['bin_return']

    # prune only lowest-impact
    to_prune = ['lagged_after_close_compound_mean','vol_5','trading_compound_count']
    X_train = X_train.drop(columns=[c for c in to_prune if c in X_train], errors='ignore')
    X_test  = X_test.drop(columns=[c for c in to_prune if c in X_test],  errors='ignore')

    # --- NEW: scale features per fold ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # train models
    lin = LinearRegression().fit(X_train, y_train)
    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    ).fit(X_train, y_train)

    # ensemble predict with fixed weights
    pred = 0.73 * lin.predict(X_test) + 0.27 * xgb.predict(X_test)

    # eval
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"Fold {i+1} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

    all_rmse.append(rmse)
    all_r2.append(r2)

# summary
print("\nBacktest average:")
print(f"RMSE: {np.mean(all_rmse):.5f} ± {np.std(all_rmse):.5f}")
print(f"R²:   {np.mean(all_r2):.5f} ± {np.std(all_r2):.5f}")
