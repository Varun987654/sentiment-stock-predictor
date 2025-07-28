# src/new_linear.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)


df['bin_return'] = df['bin_return'].shift(-1)
df.dropna(subset=['bin_return'], inplace=True)


# 2. Encode categorical/time columns (same as before)
le_ticker        = LabelEncoder(); df['ticker_id']  = le_ticker.fit_transform(df['ticker'])
le_sess          = LabelEncoder(); df['session_id'] = le_sess.fit_transform(df['session'])
times = pd.to_datetime(df['bin_end_time'], utc=True)
df['hour']           = times.dt.hour
df['minute']         = times.dt.minute
df['mins_since_mid'] = df['hour'] * 60 + df['minute']
dates = pd.to_datetime(df['date'])
df['day']        = dates.dt.day
df['month']      = dates.dt.month
df['is_weekend'] = (dates.dt.weekday >= 5).astype(int)

# 3. Prepare features & target
target_col = 'bin_return'
split_col  = 'split'
drop_cols  = ['ticker','session','bin_end_time','date', target_col, split_col]
X = df.drop(columns=drop_cols)
y = df[target_col]
splits = df[split_col]

# ensure numeric only
X = X.select_dtypes(include=[np.number, 'boolean'])

# 4. Split
X_train, y_train = X[splits=='train'], y[splits=='train']
X_val,   y_val   = X[splits=='val'],   y[splits=='val']
X_test,  y_test  = X[splits=='test'],  y[splits=='test']

# 5. Train linear model
lin = LinearRegression()
lin.fit(X_train, y_train)

# 6. Evaluate
def eval_print(name, y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:12s} → RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

print("\n== Linear Regression Performance ==")
eval_print("Validation", lin.predict(X_val), y_val)
eval_print("Test",       lin.predict(X_test), y_test)
