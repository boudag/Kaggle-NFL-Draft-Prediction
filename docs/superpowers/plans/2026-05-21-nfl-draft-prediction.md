# NFL Draft Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance ML pipeline to predict NFL Draft status using an ensemble of GBDTs with advanced imputation and feature engineering.

**Architecture:** Modular pipeline with Iterative Imputation, Target Encoding, and a weighted ensemble of XGBoost, LightGBM, and CatBoost, optimized via Optuna.

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, Category Encoders.

---

### Task 1: Environment & Data Loading

**Files:**
- Create: `src/config.py`
- Create: `src/data_loader.py`

- [ ] **Step 1: Create config file for paths and constants**
```python
import os

DATA_DIR = '/mnt/d/asu/GCI/competition_data/competition/input'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

RANDOM_STATE = 42
N_FOLDS = 10
```

- [ ] **Step 2: Create data loader**
```python
import pandas as pd
from src.config import TRAIN_PATH, TEST_PATH

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test
```

### Task 2: Preprocessing & Feature Engineering

**Files:**
- Create: `src/features.py`

- [ ] **Step 1: Implement Feature Engineering**
```python
import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    # Required Features
    df['Power_Factor'] = df['Weight'] * df['Vertical_Jump']
    df['Speed_to_Size_Ratio'] = df['Sprint_40yd'] / df['Weight']
    
    # Additional Features
    df['Total_Jump'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['Agility_Score'] = df['Agility_3cone'] + df['Shuttle']
    # Height is usually in inches or cm, let's assume it's scaled correctly for GBDT
    
    return df
```

- [ ] **Step 2: Implement Preprocessing Pipeline (Imputation & Encoding)**
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

def get_preprocessor(cat_cols):
    return {
        'imputer': IterativeImputer(random_state=42, max_iter=10),
        'encoder': TargetEncoder(cols=cat_cols, smoothing=10),
        'scaler': StandardScaler()
    }
```

### Task 3: Model Definitions & Training Loop

**Files:**
- Create: `src/models.py`
- Create: `src/train_utils.py`

- [ ] **Step 1: Define models**
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models(params_xgb=None, params_lgbm=None, params_cat=None):
    xgb = XGBClassifier(**(params_xgb or {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42}))
    lgbm = LGBMClassifier(**(params_lgbm or {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42}))
    cat = CatBoostClassifier(**(params_cat or {'n_estimators': 1000, 'learning_rate': 0.05, 'random_state': 42, 'verbose': 0}))
    return {'xgb': xgb, 'lgbm': lgbm, 'cat': cat}
```

- [ ] **Step 2: Implement Stratified K-Fold CV**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def train_and_validate(X, y, model, n_folds=10):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    oof_probs = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        scores.append(roc_auc_score(y_val, probs))
        
    return np.mean(scores), oof_probs
```

### Task 4: Optuna Optimization

**Files:**
- Create: `src/optimize.py`

- [ ] **Step 1: Implement Optuna study for XGBoost**
```python
import optuna

def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = XGBClassifier(**params, random_state=42)
    score, _ = train_and_validate(X, y, model, n_folds=5)
    return score
```

### Task 5: Main Execution & Ensembling

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement main loop with ensembling**
```python
from src.data_loader import load_data
from src.features import engineer_features
from src.train_utils import train_and_validate
# ... imports ...

def main():
    train, test = load_data()
    # Preprocess, Encode, Impute, Engineer
    # Optimize models
    # Train final ensemble
    # Predict on test
    # Save submission
```

### Task 6: Final Verification & Submission

- [ ] **Step 1: Run full pipeline**
Run: `PYTHONPATH=~/lib:. python3 main.py`
Expected: ROC AUC > 0.80 (target dependent) and `submission.csv` generated.
