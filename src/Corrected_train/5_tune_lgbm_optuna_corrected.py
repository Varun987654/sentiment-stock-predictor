# src/Corrected_train/5_tune_lgbm_optuna_corrected.py

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
import joblib
import os
import optuna

# --- CONFIG ---
INPUT_CSV = "data/06_Features_corrected/merged_features_split_corrected_final.csv"
MODEL_DIR = "models/corrected"
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

print(f"\nStarting hyperparameter tuning for LightGBM on {len(FEATURES)} features...")

# --- 3. OPTUNA OBJECTIVE FUNCTION ---
# This function defines a single "trial" where Optuna chooses a set of hyperparameters,
# trains a model, and returns its performance score.
def objective(trial):
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    return r2

# --- 4. RUN THE TUNING STUDY ---
# We tell Optuna to run the 'objective' function many times, trying to maximize the R² score.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100) # Run 100 trials to search for the best parameters

# --- 5. PRINT RESULTS & SAVE BEST MODEL ---
print("\n== Optuna Hyperparameter Tuning Complete ==")
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
best_trial = study.best_trial

print(f"  Value (R²): {best_trial.value:.5f}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Retrain the final model with the best parameters found
best_params = best_trial.params
final_model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1000, random_state=42, n_jobs=-1, **best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

joblib.dump(final_model, os.path.join(MODEL_DIR, "lgbm_tuned_corrected.pkl"))
print(f"\nSaved final tuned LightGBM model to {MODEL_DIR}")