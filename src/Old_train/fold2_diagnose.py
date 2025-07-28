# src/fold2_diagnose.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load & preprocess (same as backtest_simple)
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data/06_features/merged_features_split.csv')
df = pd.read_csv(csv_path)

# encode & time features
df['ticker_id']  = LabelEncoder().fit_transform(df['ticker'])
df['session_id'] = LabelEncoder().fit_transform(df['session'])
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
df = df.dropna(subset=[
    'roll_ret_3','roll_ret_5','vol_3','vol_5',
    'roll_sent_3','roll_sent_5','int_ret3_sent'
])

# determine fold2 dates
unique_dates = sorted(df['date'].unique())
fold_size    = len(unique_dates) // 5
fold2_dates  = unique_dates[fold_size:2*fold_size]

# split
train_df = df[~df['date'].isin(fold2_dates)]
test_df  = df[df['date'].isin(fold2_dates)]

# build X/y
drop_cols = ['ticker','session','bin_end_time','date','bin_return','split']
X_train = train_df.drop(columns=drop_cols).select_dtypes(include=[np.number,'boolean'])
y_train = train_df['bin_return']
X_test  = test_df.drop(columns=drop_cols).select_dtypes(include=[np.number,'boolean'])
y_test  = test_df['bin_return']

# prune low‑impact
to_prune = ['lagged_after_close_compound_mean','vol_5','trading_compound_count']
X_train = X_train.drop(columns=[c for c in to_prune if c in X_train], errors='ignore')
X_test  = X_test.drop(columns=[c for c in to_prune if c in X_test],  errors='ignore')

# train
lin = LinearRegression().fit(X_train, y_train)
xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50, max_depth=5, learning_rate=0.01,
    subsample=0.8, colsample_bytree=1.0, min_child_weight=3,
    random_state=42, n_jobs=-1
).fit(X_train, y_train)

# predict & eval
pred_lin = lin.predict(X_test)
pred_xgb = xgb.predict(X_test)
pred_ens = 0.73*pred_lin + 0.27*pred_xgb

rmse = np.sqrt(mean_squared_error(y_test, pred_ens))
r2   = r2_score(y_test, pred_ens)

print("Fold2 Diagnostics")
print("-----------------")
print(f"Test size: {len(X_test)} rows")
print(f"Dates  : {fold2_dates[0]} to {fold2_dates[-1]}")
print(f"RMSE   : {rmse:.5f}")
print(f"R²     : {r2:.5f}\n")

# error distribution
errors = y_test - pred_ens
print("Error quantiles:")
print(errors.quantile([0,0.25,0.5,0.75,1.0]).to_string(), "\n")

# feature distribution comparison
print("Feature means (train vs fold2 test):")
feat_means = pd.DataFrame({
    'feature': X_train.columns,
    'train_mean': X_train.mean().values,
    'test_mean' : X_test.mean().values
})
print(feat_means.sort_values('test_mean', key=lambda s: abs(s - feat_means.train_mean)).head(10).to_string(index=False))
