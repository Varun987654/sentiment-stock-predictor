import os
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb

from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 1. Load data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_Features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# 2. Inject momentum features
df['raw_return'] = (df['close'] - df['open']) / df['open']
df['mom_5']      = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(5))
df = df.dropna(subset=['raw_return', 'mom_5'])  # drop first 5 rows per ticker

# 3. Merge VIX & IRX
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

# 4. Encode categoricals
df['ticker_id']  = LabelEncoder().fit_transform(df['ticker'])
df['session_id'] = LabelEncoder().fit_transform(df['session'])

# 5. Time features
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day']        = dates.dt.day
df['month']      = dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)

# 6. Rolling/window features
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

# 7. Prepare features & target
# List of known low‑variance features to drop
low_var_feats = [
    'pre_open_compound_mean',
    'lagged_pre_open_compound_mean',
    'after_close_compound_std',
    'lagged_trading_compound_mean',
    'trading_compound_mean',
    'lagged_after_close_compound_mean',
    'after_close_compound_mean',
    'pre_open_compound_std'
]
# Control columns to exclude
exclude = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date'] + low_var_feats
X = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])
y = df['bin_return']
splits = df['split']

# 8. Split into train+val and test
X_trainval = X[splits != 'test']
y_trainval = y[splits != 'test']
X_test     = X[splits == 'test']
y_test     = y[splits == 'test']

# 9. Scale features
scaler = StandardScaler().fit(X_trainval)
X_trainval = scaler.transform(X_trainval)
X_test     = scaler.transform(X_test)

# 10. LightGBM CV to find best number of rounds
train_data = lgb.Dataset(X_trainval, label=y_trainval)
tscv       = TimeSeriesSplit(n_splits=5)

cv_results = lgb.cv(
    params={
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 7,
        'min_child_samples': 10,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0
    },
    train_set=train_data,
    num_boost_round=1000,
    folds=tscv,
    stratified=False,
    seed=42
)

# pick best round
metric_key = next(k for k in cv_results if k.endswith('-mean'))
best_rounds = int(np.argmin(cv_results[metric_key]) + 1)
print(f"CV suggests best num_boost_rounds = {best_rounds} (metric: {metric_key})")

# 11. Retrain and evaluate
model = LGBMRegressor(
    n_estimators=best_rounds,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=7,
    min_child_samples=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_trainval, y_trainval)

pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)
print(f"LightGBM CV-tuned → rounds={best_rounds}, RMSE: {rmse:.5f}, R²: {r2:.5f}")
