# src/Corrected_train/Phase2_6_feature_selection.py

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/advanced_features.csv"
MODEL_DIR = "models/corrected_advanced"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
df_train = df[df["split"] == "train"].copy()
df_val = df[df["split"] == "val"].copy()
df_test = df[df["split"] == "test"].copy()

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Convert ticker to a numerical category
df_train['ticker'] = pd.Categorical(df_train['ticker']).codes
df_val['ticker'] = pd.Categorical(df_val['ticker']).codes
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val = df_val[FEATURES]
y_val = df_val[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

# --- 3. TRAIN A MODEL ON ALL FEATURES TO GET IMPORTANCE ---
print("\nTraining initial model to determine feature importance...")
model_for_importance = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, # No need for a large number here
    random_state=42,
    n_jobs=-1
)
model_for_importance.fit(X_train, y_train)

# --- 4. EXTRACT AND DISPLAY TOP FEATURES ---
feature_importances = pd.Series(model_for_importance.feature_importances_, index=FEATURES).sort_values(ascending=False)
TOP_N_FEATURES = 10
top_features = feature_importances.head(TOP_N_FEATURES).index.tolist()

print(f"\n--- Top {TOP_N_FEATURES} Most Important Features ---")
print(feature_importances.head(TOP_N_FEATURES))
print("\nRetraining a new model on only these features...")

# --- 5. RETRAIN A NEW MODEL ON ONLY THE TOP FEATURES ---
X_train_top = df_train[top_features]
X_val_top = df_val[top_features]
X_test_top = df_test[top_features]

final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)
final_model.fit(
    X_train_top, y_train,
    eval_set=[(X_val_top, y_val)],
    verbose=False
)

# --- 6. EVALUATE THE NEW, SIMPLER MODEL ---
def evaluate(X, y, name):
    preds = final_model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== XGBoost Performance (Top 10 Features) ==")
evaluate(X_val_top, y_val, "Validation")
evaluate(X_test_top, y_test, "Test")

joblib.dump(final_model, os.path.join(MODEL_DIR, "xgb_top10_features.pkl"))
print(f"\nSaved final Top 10 Feature XGBoost model to {MODEL_DIR}")