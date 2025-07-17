import pandas as pd
import numpy as np

# 1) Load
df = pd.read_csv('data/06_features/merged_features_split.csv')

# 2) Identify numeric features (drop only real control columns)
exclude = ['ticker','session','bin_end_time','date','split']
num = df.drop(columns=exclude, errors='ignore').select_dtypes(include=[np.number, 'boolean'])

# 3) Compute variances
variances = num.var()

# 4) Print lowest & highest
print("Lowest‑variance features:")
print(variances.nsmallest(10))
print("\nHighest‑variance features:")
print(variances.nlargest(10))
