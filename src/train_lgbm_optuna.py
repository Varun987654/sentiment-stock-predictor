import os
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import optuna

from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ——————————————————————————————————————————————
# 1) Load & preprocess (same as your ES script)
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
csv_path     = os.path.join(project_root, 'data', '06_features', 'merged_features_split.csv')
df = pd.read_csv(csv_path)

# […copy over your VIX/IRX merge, encodings, time features, rolling features…]
# Drop NaNs, prepare X, y, splits… then scale exactly as in train_lgbm_es.py
exclude = ['ticker','session','bin_end_time','date','bin_return','split','date_dt','Date']
to_drop = [c for c in exclude if c in df.columns]
X = df.drop(columns=to_drop).select_dtypes(include=[np.number, 'boolean'])
y = df['bin_return']
splits = df['split']

X_trainval = X[splits != 'test']
y_trainval = y[splits != 'test']
X_test     = X[splits == 'test']
y_test     = y[splits == 'test']

scaler = StandardScaler().fit(X_trainval)
X_trainval = scaler.transform(X_trainval)
X_test     = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

# ——————————————————————————————————————————————
# 2) Optuna objective WITHOUT early_stopping_rounds
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbosity': -1
    }
    # run full CV
    cv_hist = lgb.cv(
        params,
        lgb.Dataset(X_trainval, label=y_trainval),
        folds=tscv,
        stratified=False,
        num_boost_round=1000,
        seed=42
    )
    # pick the minimal mean RMSE
    key = next(k for k in cv_hist if k.endswith('-mean'))
    best_rmse = min(cv_hist[key])
    return best_rmse

# ——————————————————————————————————————————————
# 3) Run the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # you can bump to 50–100 trials

print("Best trial parameters:")
print(study.best_trial.params)

# ——————————————————————————————————————————————
# 4) Retrain final model on full train+val
best_params = study.best_trial.params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1
})
# rerun CV to find best rounds
cv_final = lgb.cv(
    best_params,
    lgb.Dataset(X_trainval, label=y_trainval),
    folds=tscv,
    stratified=False,
    num_boost_round=1000,
    seed=42
)
round_key = next(k for k in cv_final if k.endswith('-mean'))
best_rounds = int(np.argmin(cv_final[round_key]) + 1)
print(f"Retrained with best_rounds = {best_rounds}")

model = LGBMRegressor(**best_params, n_estimators=best_rounds, random_state=42, n_jobs=-1)
model.fit(X_trainval, y_trainval)

# ——————————————————————————————————————————————
# 5) Final Evaluation
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2   = r2_score(y_test, preds)
print(f"OPTUNA‐Tuned LightGBM → RMSE: {rmse:.5f}, R²: {r2:.5f}")
