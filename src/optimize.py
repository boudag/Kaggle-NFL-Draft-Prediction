import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.train_utils import train_and_validate

def objective_xgb(trial, X, y, cat_cols, num_cols_to_clip):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'enable_categorical': True,
        'tree_method': 'hist',
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    def factory(): return XGBClassifier(**params)
    score, _, _ = train_and_validate(X, y, factory, n_folds=5, cat_cols=cat_cols, num_cols_to_clip=num_cols_to_clip)
    return score

def objective_lgbm(trial, X, y, cat_cols, num_cols_to_clip):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbose': -1
    }
    def factory(): return LGBMClassifier(**params)
    score, _, _ = train_and_validate(X, y, factory, n_folds=5, cat_cols=cat_cols, num_cols_to_clip=num_cols_to_clip)
    return score

def objective_cat(trial, X, y, cat_cols, num_cols_to_clip):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'verbose': 0,
        'eval_metric': 'AUC',
        'cat_features': cat_cols
    }
    def factory(): return CatBoostClassifier(**params)
    score, _, _ = train_and_validate(X, y, factory, n_folds=5, cat_cols=cat_cols, num_cols_to_clip=num_cols_to_clip)
    return score
