from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models(params_xgb=None, params_lgbm=None, params_cat=None):
    # Default params chosen for stability
    xgb_defaults = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    lgbm_defaults = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    cat_defaults = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'random_state': 42,
        'verbose': 0,
        'eval_metric': 'AUC'
    }
    
    xgb = XGBClassifier(**(params_xgb or xgb_defaults))
    lgbm = LGBMClassifier(**(params_lgbm or lgbm_defaults))
    cat = CatBoostClassifier(**(params_cat or cat_defaults))
    
    return {'xgb': xgb, 'lgbm': lgbm, 'cat': cat}
