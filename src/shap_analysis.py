# src/shap_analysis.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import shap

# 1. Load data
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# 2. Preprocessing (same as final_train_xgb_with_rolling.py)
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
# rolling/window features
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

# 3. Build feature matrix & splits
drop_cols = ['ticker','session','bin_end_time','date','bin_return','split']
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number, 'boolean'])
y = df['bin_return']
splits = df['split']
X_trainval = X[splits != 'test']
y_trainval = y[splits != 'test']

# 4. Train tuned XGBoost
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
model.fit(X_trainval, y_trainval)

# 5. SHAP explainer
explainer = shap.Explainer(model, X_trainval)
shap_values = explainer(X_trainval)

# 6. Compute mean absolute SHAP value per feature
shap_abs = np.abs(shap_values.values)
mean_imp = np.mean(shap_abs, axis=0)
feat_imp = pd.DataFrame({
    'feature': X_trainval.columns,
    'mean_abs_shap': mean_imp
}).sort_values('mean_abs_shap', ascending=False)

# 7. Print top 20 features
print("Top 20 features by mean |SHAP value|:")
print(feat_imp.head(20).to_string(index=False))
