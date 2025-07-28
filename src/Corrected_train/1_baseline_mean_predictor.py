# src/Corrected_train/1_baseline_mean_predictor.py

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# --- CONFIG ---
# This is our final, clean, non-leaky dataset
INPUT_CSV = "data/06_Features_corrected/merged_features_split_corrected_final.csv"
MODEL_DIR = "models/corrected"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

# --- 2. "TRAIN" THE BASELINE MODEL ---
# The only "training" is to find the average return of the training set.
mean_train_return = df_train['target_return'].mean()
print(f"\nCalculated mean return from training data: {mean_train_return:.8f}")

# --- 3. "PREDICT" & EVALUATE ---
# The "prediction" is just this single mean value for every row in the validation and test sets.
y_val = df_val['target_return']
preds_val = np.full(len(y_val), mean_train_return)

y_test = df_test['target_return']
preds_test = np.full(len(y_test), mean_train_return)

def evaluate(y_true, y_preds, name):
    # --- FIX: Calculate RMSE manually for older scikit-learn versions ---
    mse = mean_squared_error(y_true, y_preds)
    rmse = np.sqrt(mse) # Take the square root of MSE to get RMSE
    r2 = r2_score(y_true, y_preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Mean Predictor Performance (Corrected Data) ==")
evaluate(y_val, preds_val, "Validation")
evaluate(y_test, preds_test, "Test")

# --- 4. SAVE THE BASELINE VALUE ---
# We save the mean value itself as our "model"
baseline_value_path = os.path.join(MODEL_DIR, "mean_predictor_value.txt")
with open(baseline_value_path, 'w') as f:
    f.write(str(mean_train_return))
print(f"\nSaved baseline mean value to {baseline_value_path}")