# src/Corrected_train/Phase2_7_tune_xgb_top10.py

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
import joblib
import os
import optuna

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/advanced_features.csv"
MODEL_DIR = "models/corrected_advanced"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. LOAD DATA ---
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
df_train = df[df["split"] == "train"].copy()
df_val = df[df["split"] == "val"].copy()

# --- 2. DEFINE FEATURES & TARGET ---
TARGET = 'target_return'
# --- Using only the Top 10 features we discovered ---
TOP_10_FEATURES = ['brent_oil', 'prev_low', 'gold_price', 'sentiment_x_volatility', 
                   'trading_compound_count_60m', 'bin_slot_number', 
                   'trading_compound_count_180m', 'mean_pre_open', 'volume', 
                   'trading_compound_mean_180m']

# Preprocess data
df_train['ticker'] = pd.Categorical(df_train['ticker']).codes
df_val['ticker'] = pd.Categorical(df_val['ticker']).codes

X_train = df_train[TOP_10_FEATURES]
y_train = df_train[TARGET]
X_val = df_val[TOP_10_FEATURES]
y_val = df_val[TARGET]

print(f"\nStarting hyperparameter tuning for XGBoost on the TOP 10 features...")

# --- 3. OPTUNA OBJECTIVE FUNCTION ---
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 20.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    return r2

# --- 4. RUN THE TUNING STUDY ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# --- 5. PRINT RESULTS & SAVE BEST MODEL ---
print("\n== Optuna Hyperparameter Tuning Complete (Top 10 Features) ==")
best_trial = study.best_trial
print(f"Best R² on Validation Set: {best_trial.value:.5f}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

best_params = best_trial.params
final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=42, n_jobs=-1, **best_params)

try:
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
except TypeError:
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

joblib.dump(final_model, os.path.join(MODEL_DIR, "xgb_tuned_top10.pkl"))
print(f"\nSaved final tuned (Top 10) XGBoost model to {MODEL_DIR}")