import pandas as pd
import numpy as np
import optuna
from src.data_loader import load_data
from src.features import engineer_features
from src.preprocess import PreprocessingPipeline
from src.models import get_models
from src.train_utils import train_and_validate
from src.optimize import objective_xgb, objective_lgbm, objective_cat
from src.config import SUBMISSION_PATH
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def main():
    print("Loading data...")
    train, test = load_data()
    
    print("Engineering features...")
    train = engineer_features(train)
    test = engineer_features(test)
    
    X = train.drop(['Drafted', 'Id'], axis=1)
    y = train['Drafted']
    X_test_raw = test.drop(['Id'], axis=1)
    
    cat_cols = ['School', 'Position', 'Player_Type', 'Position_Type', 'Year']
    
    print("Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y, cat_cols), n_trials=10)
    best_params_xgb = study_xgb.best_params
    
    print("Optimizing LightGBM...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y, cat_cols), n_trials=10)
    best_params_lgbm = study_lgbm.best_params
    
    print("Optimizing CatBoost...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda trial: objective_cat(trial, X, y, cat_cols), n_trials=10)
    best_params_cat = study_cat.best_params
    
    print("Final training and ensembling...")
    # Get OOF and test predictions for each model
    results = {}
    test_probs = {}
    
    for name, factory_fn, params in [
        ('xgb', lambda p: XGBClassifier(**p, random_state=42), best_params_xgb),
        ('lgbm', lambda p: LGBMClassifier(**p, random_state=42, verbose=-1), best_params_lgbm),
        ('cat', lambda p: CatBoostClassifier(**p, random_state=42, verbose=0), best_params_cat)
    ]:
        print(f"Training {name} with best params...")
        score, oof = train_and_validate(X, y, lambda: factory_fn(params), n_folds=10, cat_cols=cat_cols)
        results[name] = {'score': score, 'oof': oof}
        
        # Fit on full data and predict test
        pipeline = PreprocessingPipeline(cat_cols)
        X_full = pipeline.fit_transform(X, y)
        X_test = pipeline.transform(X_test_raw)
        
        final_model = factory_fn(params)
        final_model.fit(X_full, y)
        test_probs[name] = final_model.predict_proba(X_test)[:, 1]
        print(f"{name} CV Score: {score}")

    # Weighted ensemble based on CV scores (simple normalization)
    scores = np.array([results[m]['score'] for m in results])
    weights = scores / scores.sum()
    print(f"Ensemble weights: {dict(zip(results.keys(), weights))}")
    
    final_test_probs = np.zeros(len(test))
    for i, name in enumerate(results.keys()):
        final_test_probs += test_probs[name] * weights[i]
        
    print("Generating submission...")
    submission = pd.DataFrame({
        'Id': test['Id'],
        'Drafted': final_test_probs
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
