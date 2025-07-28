# src/train_baseline.py

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Load data ===
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')

df = pd.read_csv(csv_path)

# === 2. Identify target and split columns ===
target_col = 'bin_return'
split_col = 'split'

# === 3. Separate features and target ===
X = df.drop(columns=[target_col, split_col])
y = df[target_col]
splits = df[split_col]

# === 4. Keep only numeric features ===
# Select all columns of numeric dtype (int, float, bool)
X = X.select_dtypes(include=[np.number, 'boolean'])

# === 5. Train/Validation/Test split ===
X_train = X[splits == 'train']
y_train = y[splits == 'train']

X_val = X[splits == 'val']
y_val = y[splits == 'val']

X_test = X[splits == 'test']
y_test = y[splits == 'test']

# === 6. Train baseline XGBoost ===
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluate ===
def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} → RMSE: {rmse:.5f}, MAE: {mae:.5f}, R²: {r2:.5f}")

# Print validation and test metrics
evaluate("Validation", y_val, model.predict(X_val))
evaluate("Test",       y_test,  model.predict(X_test))

# === 8. Benchmark: Mean‑Predictor ===
mean_return = y_train.mean()

# Constant predictions
y_val_mean  = [mean_return] * len(y_val)
y_test_mean = [mean_return] * len(y_test)

print("\n--- Mean-Predictor Performance ---")
evaluate("Validation (mean)", y_val,  y_val_mean)
evaluate("Test       (mean)", y_test, y_test_mean)
