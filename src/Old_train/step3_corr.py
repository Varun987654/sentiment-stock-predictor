import pandas as pd
import numpy as np

# 1) Load
df = pd.read_csv('data/06_features/merged_features_split.csv')

# 2) Recreate X exactly as in train_lgbm_es.py
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
exclude = ['ticker','session','bin_end_time','date','bin_return','split'] + low_var_feats
X = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])

# 3) Compute correlations
corrs = X.corrwith(df['bin_return']).abs().sort_values(ascending=False)

# 4) Print the top 10 features by absolute correlation
print("Top 10 features by |corr| with bin_return:")
print(corrs.head(10))
