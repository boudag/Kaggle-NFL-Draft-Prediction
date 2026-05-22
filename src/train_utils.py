from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from src.preprocess import PreprocessingPipeline

def train_and_validate(X, y, model_factory, n_folds=10, cat_cols=None, num_cols_to_clip=None):
    # No random_state for seed-free
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    scores = []
    oof_probs = np.zeros(len(X))
    fitted_models = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Get a fresh model
        model = model_factory()
        model_name = type(model).__name__.lower()
        is_linear = "logisticregression" in model_name
        
        # Preprocess inside fold to avoid leakage
        pipeline = PreprocessingPipeline(cat_cols, num_cols_to_clip=num_cols_to_clip, impute_and_scale=is_linear)
        X_train = pipeline.fit_transform(X_train_raw, y_train)
        X_val = pipeline.transform(X_val_raw)
        
        fit_kwargs = {}
        if "xgb" in model_name:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['verbose'] = False
        elif "lgbm" in model_name:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
        elif "catboost" in model_name:
            fit_kwargs['eval_set'] = (X_val, y_val)
            fit_kwargs['verbose'] = False
            
        model.fit(X_train, y_train, **fit_kwargs)
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        score = roc_auc_score(y_val, probs)
        scores.append(score)
        
        fitted_models.append((pipeline, model))
        
    return np.mean(scores), oof_probs, fitted_models
