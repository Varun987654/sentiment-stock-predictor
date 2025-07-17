# src/train_with_encoding.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Load data ===
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')

df = pd.read_csv(csv_path)

# === 2. Encode object columns ===

# 2.a Ticker → integer ID
le_ticker        = LabelEncoder()
df['ticker_id']  = le_ticker.fit_transform(df['ticker'])

# 2.b Session → integer ID
le_sess          = LabelEncoder()
df['session_id'] = le_sess.fit_transform(df['session'])

# 2.c Bin end time → parse ISO8601 timestamp, then extract time components
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']

# 2.d Date → day, month, is_weekend
dates             = pd.to_datetime(df['date'])
df['day']         = dates.dt.day
df['month']       = dates.dt.month
df['is_weekend']  = (dates.dt.weekday >= 5).astype(int)

# === 3. Prepare feature matrix & target ===
target_col = 'bin_return'
split_col  = 'split'

# Drop original object columns now encoded, plus target & split
drop_cols = ['ticker', 'session', 'bin_end_time', 'date', target_col, split_col]
X = df.drop(columns=drop_cols)
y = df[target_col]
splits = df[split_col]

# === 4. Ensure numeric-only features ===
X = X.select_dtypes(include=[np.number, 'boolean'])

# === 5. Split into train/val/test ===
X_train = X[splits == 'train']
y_train = y[splits == 'train']

X_val = X[splits == 'val']
y_val = y[splits == 'val']

X_test = X[splits == 'test']
y_test = y[splits == 'test']

# === 6. Train XGBoost ===
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluation helper ===
def evaluate(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:12s} → RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# === 8. Evaluate on validation & test ===
print("\n=== Model Performance ===")
evaluate("Validation", y_val,  model.predict(X_val))
evaluate("Test",       y_test, model.predict(X_test))
