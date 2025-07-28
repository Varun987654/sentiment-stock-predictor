# src/Corrected_train/2_train_linear_corrected.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/merged_features_split_corrected_final.csv"
MODEL_DIR = "models/corrected"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

# --- 2. DEFINE FEATURES & TARGET ---
# We exclude identifiers and the target itself from the feature list
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val = df_val[FEATURES]
y_val = df_val[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

print(f"\nTraining model on {len(FEATURES)} features...")

# --- 3. TRAIN LINEAR REGRESSION ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. EVALUATE ---
def evaluate(X, y, name):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse) # Take the square root of MSE to get RMSE
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== Linear Regression Performance (Corrected Data) ==")
evaluate(X_val, y_val, "Validation")
evaluate(X_test, y_test, "Test")

# --- 5. SAVE MODEL ---
joblib.dump(model, os.path.join(MODEL_DIR, "linear_regression_corrected.pkl"))
print(f"\nSaved corrected Linear Regression model to {MODEL_DIR}")