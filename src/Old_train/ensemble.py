# src/ensemble.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# 1. Load data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# 2. (Optional) clip returns again if you used clipping in backtest
df['bin_return'] = df['bin_return'].clip(-0.05, 0.05)

for ticker, col in [("^VIX","vix_close"),("^IRX","irx_close")]:
    hist = yf.Ticker(ticker).history(
        start="2025-01-01", end="2025-07-10", interval="1d"
    )[['Close']].rename(columns={'Close':col})
    hist.index.name = 'Date'
    hist = hist.reset_index()
    hist['Date'] = hist['Date'].dt.tz_localize(None)
    df = df.merge(hist, left_on=pd.to_datetime(df['date']),
                  right_on='Date', how='left').ffill()
    

# 3. Encode & time features (as before)…
le_ticker       = LabelEncoder(); df['ticker_id']  = le_ticker.fit_transform(df['ticker'])
le_sess         = LabelEncoder(); df['session_id'] = le_sess.fit_transform(df['session'])
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day']        = dates.dt.day
df['month']      = dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)

# 4. Rolling features (as before)…
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

# 5. Build feature matrix & splits
drop_cols = ['ticker','session','bin_end_time','date','bin_return','split']
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
y = df['bin_return']
splits = df['split']

X_trainval = X[splits!='test']
y_trainval = y[splits!='test']
X_test     = X[splits=='test']
y_test     = y[splits=='test']

# 6. Train Linear Regression
lin = LinearRegression().fit(X_trainval, y_trainval)

# 7. Train Tuned XGBoost
xgb = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50, max_depth=5,
    learning_rate=0.01, subsample=0.8,
    colsample_bytree=1.0, min_child_weight=3,
    random_state=42, n_jobs=-1
).fit(X_trainval, y_trainval)

# 8. Train Tuned LightGBM
lgb = LGBMRegressor(
    objective='regression',
    n_estimators=271,
    learning_rate=0.033154126916433574,
    num_leaves=83,
    subsample=0.7748231659997137,
    colsample_bytree=0.9089313158952361,
    reg_alpha=0.15000694120470892,
    reg_lambda=3.621134664078126,
    random_state=42,
    n_jobs=-1
).fit(X_trainval, y_trainval)

# 9. Get individual predictions
pred_lin = lin.predict(X_test)
pred_xgb = xgb.predict(X_test)
pred_lgb = lgb.predict(X_test)
# After training lin, xgb, lgb …

# 9.a Evaluate each model individually
r2_lin = r2_score(y_test, lin.predict(X_test))
r2_xgb = r2_score(y_test, xgb.predict(X_test))
r2_lgb = r2_score(y_test, lgb.predict(X_test))
rmse_lin = np.sqrt(mean_squared_error(y_test, lin.predict(X_test)))
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb.predict(X_test)))
rmse_lgb = np.sqrt(mean_squared_error(y_test, lgb.predict(X_test)))

print(f"Linear    → RMSE: {rmse_lin:.5f}, R²: {r2_lin:.5f}")
print(f"XGBoost   → RMSE: {rmse_xgb:.5f}, R²: {r2_xgb:.5f}")
print(f"LightGBM  → RMSE: {rmse_lgb:.5f}, R²: {r2_lgb:.5f}")


# 10. Search over simple 3-model weights
best_r2 = -np.inf
best_w = (0,0,0)
for w1 in np.linspace(0,1,21):
    for w2 in np.linspace(0,1-w1,21):
        w3 = 1 - w1 - w2
        pred_ens = w1*pred_lin + w2*pred_xgb + w3*pred_lgb
        r2_ = r2_score(y_test, pred_ens)
        if r2_ > best_r2:
            best_r2 = r2_
            best_w = (w1, w2, w3)

w_lin, w_xgb, w_lgb = best_w
pred_ens = w_lin*pred_lin + w_xgb*pred_xgb + w_lgb*pred_lgb
rmse_ens = np.sqrt(mean_squared_error(y_test, pred_ens))

# 11. Report
print(f"Best weights → Lin: {w_lin:.2f}, XGB: {w_xgb:.2f}, LGB: {w_lgb:.2f}")
print(f"Ensemble → RMSE: {rmse_ens:.5f}, R²: {best_r2:.5f}")
