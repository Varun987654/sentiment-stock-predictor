# src/stack_ensemble.py

import os
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# 2. Clip extreme returns
df['bin_return'] = df['bin_return'].clip(-0.05, 0.05)

# 3. Fetch VIX
vix = yf.Ticker("^VIX") \
    .history(start="2025-01-01", end="2025-07-10", interval="1d")[['Close']] \
    .rename(columns={'Close':'vix_close'})
vix.index.name = 'Date'
vix = vix.reset_index()
vix['Date'] = vix['Date'].dt.tz_localize(None)

# 4. Merge VIX into main DataFrame
df['date_dt'] = pd.to_datetime(df['date'])
df = df.merge(vix, left_on='date_dt', right_on='Date', how='left').ffill()

# 5. Encode categorical features
df['ticker_id']  = LabelEncoder().fit_transform(df['ticker'])
df['session_id'] = LabelEncoder().fit_transform(df['session'])

# 6. Create time‑based features
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day']        = dates.dt.day
df['month']      = dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)

# 7. Rolling/window features
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

# 8. Prepare features & target
drop_cols = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date']
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
y = df['bin_return']
splits = df['split']

X_trainval = X[splits!='test']
y_trainval = y[splits!='test']
X_test     = X[splits=='test']
y_test     = y[splits=='test']

# 9. Scale features
scaler = StandardScaler().fit(X_trainval)
X_trainval = scaler.transform(X_trainval)
X_test     = scaler.transform(X_test)

# 10. Define base learners
base_learners = [
    ('lin', LinearRegression()),
    ('xgb', XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    ))
]

# 11. Create stacking ensemble
stack = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(alpha=1.0),
    cv=3,
    passthrough=True
)

# 12. Train
stack.fit(X_trainval, y_trainval)

# 13. Predict & evaluate
pred = stack.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)

print(f"Stacking Ensemble → RMSE: {rmse:.5f}, R²: {r2:.5f}")
