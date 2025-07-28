import pandas as pd
import numpy as np

# Load full df
df = pd.read_csv('data/06_features/merged_features_split.csv')

# Apply the same drops as in train_lgbm_es.py
low_var_feats = [
    'lagged_return',
    'pre_open_compound_mean',
    'lagged_pre_open_compound_mean',
    'after_close_compound_std',
    'lagged_trading_compound_mean',
    'trading_compound_mean',
    'lagged_after_close_compound_mean',
    'after_close_compound_mean',
    'pre_open_compound_std'
]
exclude = ['ticker','session','bin_end_time','date','bin_return','split'] + low_var_feats
X = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])

# Only training rows
X_trainval = X[df['split'] != 'test']

# 1) Shape
print("X_trainval shape:", X_trainval.shape)

# 2) Column list
print("Features:", list(X_trainval.columns))

# 3) Sample values
print("\nFirst 5 rows:\n", X_trainval.head())
