# src/final_train_xgb_with_rolling.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data and encoding (same as train_xgb_with_rolling.py)
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# Encode categorical/time
le_ticker       = LabelEncoder(); df['ticker_id']   = le_ticker.fit_transform(df['ticker'])
le_sess         = LabelEncoder(); df['session_id']  = le_sess.fit_transform(df['session'])
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates            = pd.to_datetime(df['date'])
df['day']         = dates.dt.day
df['month']       = dates.dt.month
df['is_weekend']  = (dates.dt.weekday >= 5).astype(int)

# Rolling/window features
df['roll_ret_3']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_ret_5']    = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).mean())
df['vol_3']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).std())
df['vol_5']         = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).std())
df['roll_sent_3']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_sent_5']   = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(5).mean())
df['int_ret3_sent'] = df['roll_ret_3'] * df['trading_compound_mean']

# Drop NaNs from rolling
df = df.dropna(subset=[
    'roll_ret_3','roll_ret_5','vol_3','vol_5',
    'roll_sent_3','roll_sent_5','int_ret3_sent'
])

# Build features & splits
target_col = 'bin_return'
split_col  = 'split'
drop_cols  = ['ticker','session','bin_end_time','date', target_col, split_col]
X_all = df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
y_all = df[target_col]
splits = df[split_col]

# 2. Combine train + val for final training
mask = splits.isin(['train','val'])
X_trainval = X_all[mask]
y_trainval = y_all[mask]

# 3. Isolate test set
X_test = X_all[splits=='test']
y_test = y_all[splits=='test']

# 4. Initialize XGBoost with best hyperparameters
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)

# 5. Train on train+val
model.fit(X_trainval, y_trainval)

# 6. Predict and evaluate on test
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=== Final Rolling‑XGBoost Performance on TEST Set ===")
print(f"RMSE: {rmse:.5f}")
print(f"MAE:  {mae:.5f}")
print(f"R²:   {r2:.5f}")
