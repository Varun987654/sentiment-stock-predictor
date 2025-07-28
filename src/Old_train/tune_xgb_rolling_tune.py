# src/tune_xgb.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# 1. Load & encode data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# Encode categorical/time columns (same as before)
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

df['roll_ret_3']  = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_ret_5']  = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).mean())
df['vol_3']       = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(3).std())
df['vol_5']       = df.groupby('ticker')['bin_return'].transform(lambda x: x.shift().rolling(5).std())
df['roll_sent_3'] = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(3).mean())
df['roll_sent_5'] = df.groupby('ticker')['trading_compound_mean'].transform(lambda x: x.shift().rolling(5).mean())
df['int_ret3_sent'] = df['roll_ret_3'] * df['trading_compound_mean']

# 4. Drop rows with NaNs from rolling
df = df.dropna(subset=[
    'roll_ret_3','roll_ret_5','vol_3','vol_5',
    'roll_sent_3','roll_sent_5','int_ret3_sent'
])

# Build X, y, splits
target_col = 'bin_return'
split_col  = 'split'
drop_cols  = ['ticker','session','bin_end_time','date', target_col, split_col]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
y = df[target_col]
splits = df[split_col]

# Train‐only on the 'train' split for hyperparameter tuning
X_train = X[splits == 'train']
y_train = y[splits == 'train']

# 2. Define base model & param distribution
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators':    [50, 100, 200, 400],
    'max_depth':       [3, 5, 7, 9],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.6, 0.8, 1.0],
    'min_child_weight':[1, 3, 5]
}

# 3. Define RMSE scorer (negated for “higher is better”)
def neg_rmse(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(neg_rmse)

# 4. Setup RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=40,
    scoring=rmse_scorer,
    cv=3,
    verbose=2,
    random_state=42,
    error_score='raise'
)

# 5. Run the search
search.fit(X_train, y_train)

# 6. Report best params & CV score
print("\nBest hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nBest CV score (neg RMSE): {search.best_score_:.5f}")
