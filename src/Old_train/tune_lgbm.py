# src/tune_lgbm.py

import os
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load & preprocess (same as backtest_simple_scaled)
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data/06_features/merged_features_split.csv')
df = pd.read_csv(csv_path)

# Clip returns
df['bin_return'] = df['bin_return'].clip(-0.05, 0.05)

# Fetch & merge VIX & IRX (you already added these; just re‑use that block)
import yfinance as yf
df['date_dt'] = pd.to_datetime(df['date'])
for ticker, col in [("^VIX","vix_close"),("^IRX","irx_close")]:
    hist = yf.Ticker(ticker).history(
        start="2025-01-01", end="2025-07-10", interval="1d"
    )[['Close']].rename(columns={'Close':col})
    hist.index.name = 'Date'
    hist = hist.reset_index()
    hist['Date'] = hist['Date'].dt.tz_localize(None)
    df = df.merge(hist, left_on='date_dt', right_on='Date', how='left').ffill()

# Encode & time features (copy your code)
df['ticker_id']  = LabelEncoder().fit_transform(df['ticker'])
df['session_id'] = LabelEncoder().fit_transform(df['session'])
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour'], df['minute'] = times.dt.hour, times.dt.minute
df['mins_since_mid'] = df['hour']*60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day'], df['month'] = dates.dt.day, dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)
# Rolling features
df['roll_ret_3']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_ret_5']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).mean())
df['vol_3']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).std())
df['vol_5']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).std())
df['roll_sent_3']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_sent_5']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(5).mean())
df['int_ret3_sent'] = df['roll_ret_3'] * df['trading_compound_mean']
df = df.dropna(subset=['roll_ret_3','roll_ret_5','vol_3','vol_5','roll_sent_3','roll_sent_5','int_ret3_sent'])

# Build X/y for train+val only
drop = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date']
to_drop = [c for c in drop if c in df.columns]
X = df.drop(columns=to_drop).select_dtypes(include=[np.number,'boolean'])
y = df['bin_return']
splits = df['split']
X_trainval = X[splits!='test']
y_trainval = y[splits!='test']

# Scale
scaler = StandardScaler().fit(X_trainval)
X_trainval = scaler.transform(X_trainval)

# 2. Define Optuna objective
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
    }
    model = LGBMRegressor(**params, random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    # negative RMSE
    score = -np.mean(cross_val_score(
        model, X_trainval, y_trainval,
        cv=tscv, scoring='neg_root_mean_squared_error'
    ))
    return score

# 3. Run study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

print("Best LGBM params:")
print(study.best_params)
