# src/Corrected_train/Phase2_4_tune_xgb_advanced.py

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
FEATURES = [col for col in df.columns if col not in ['ticker', 'bin_end_time', 'date', 'split', TARGET]]

df_train['ticker'] = pd.Categorical(df_train['ticker']).codes
df_val['ticker'] = pd.Categorical(df_val['ticker']).codes

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_val = df_val[FEATURES]
y_val = df_val[TARGET]

print(f"\nStarting hyperparameter tuning for XGBoost on {len(FEATURES)} advanced features...")

# --- 3. OPTUNA OBJECTIVE FUNCTION ---
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1,
        # --- FIX: Define early stopping here, in the model's parameters ---
        'early_stopping_rounds': 50
    }

    model = xgb.XGBRegressor(**params)
    
    # Pass the evaluation set without the early_stopping_rounds keyword
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    return r2

# --- 4. RUN THE TUNING STUDY ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100) # Run 100 trials

# --- 5. PRINT RESULTS & SAVE BEST MODEL ---
print("\n== Optuna Hyperparameter Tuning Complete ==")
best_trial = study.best_trial
print(f"Best R² on Validation Set: {best_trial.value:.5f}")
print("Best hyperparameters:")
# The early_stopping_rounds will be in the params, so we can use them directly
best_params = best_trial.params

for key, value in best_params.items():
    print(f"  {key}: {value}")

# Retrain the final model with the best parameters
final_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=1000, 
    random_state=42, 
    n_jobs=-1, 
    **best_params # best_params now includes the early_stopping_rounds
)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

joblib.dump(final_model, os.path.join(MODEL_DIR, "xgb_tuned_advanced.pkl"))
print(f"\nSaved final tuned XGBoost model to {MODEL_DIR}")