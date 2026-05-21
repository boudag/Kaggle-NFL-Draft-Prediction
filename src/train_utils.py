from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from src.preprocess import PreprocessingPipeline

def train_and_validate(X, y, model_factory, n_folds=10, cat_cols=None):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    oof_probs = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Preprocess inside fold to avoid leakage
        pipeline = PreprocessingPipeline(cat_cols)
        X_train = pipeline.fit_transform(X_train_raw, y_train)
        X_val = pipeline.transform(X_val_raw)
        
        # Get a fresh model
        model = model_factory()
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        score = roc_auc_score(y_val, probs)
        scores.append(score)
        
    return np.mean(scores), oof_probs
