# src/Corrected_train/Phase2_2_train_lgbm_advanced.py

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# --- CONFIG ---
# --- Pointing to the new advanced features file ---
INPUT_CSV = "data/06_Features_corrected/advanced_features.csv"
MODEL_DIR = "models/corrected_advanced"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
# Use .copy() to avoid SettingWithCopyWarning
df_train = df[df["split"] == "train"].copy()
df_val = df[df["split"] == "val"].copy()
df_test = df[df["split"] == "test"].copy()

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

# Convert ticker to a numerical category for LightGBM
df_train['ticker'] = pd.Categorical(df_train['ticker']).codes
df_val['ticker'] = pd.Categorical(df_val['ticker']).codes
df_test['ticker'] = pd.Categorical(df_test['ticker']).codes

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val = df_val[FEATURES]
y_val = df_val[TARGET]
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

print(f"\nTraining LightGBM model on {len(FEATURES)} advanced features...")

# --- 3. TRAIN BASELINE LIGHTGBM MODEL ---
# Using the best parameters we found from our previous tuning session
best_params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.047089609789474136,
    'num_leaves': 228,
    'max_depth': 9,
    'subsample': 0.6605844672854039,
    'colsample_bytree': 0.9738917745123822,
    'reg_alpha': 0.008893783681768158,
    'reg_lambda': 0.042331032987156,
    'random_state': 42,
    'n_jobs': -1,
}

model = lgb.LGBMRegressor(**best_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# --- 4. EVALUATE ---
def evaluate(X, y, name):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"{name:<12} → RMSE: {rmse:.5f}, R²: {r2:.5f}")

print("\n== LightGBM Performance (Advanced Features) ==")
print(f"   (Best iteration: {model.best_iteration_})")
evaluate(X_val, y_val, "Validation")
evaluate(X_test, y_test, "Test")

# --- 5. SAVE MODEL ---
joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_advanced.pkl"))
print(f"\nSaved LightGBM model (trained on advanced features) to {MODEL_DIR}")