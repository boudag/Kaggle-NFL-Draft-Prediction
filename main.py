import pandas as pd
import numpy as np
import optuna
from scipy.optimize import minimize
from src.data_loader import load_data
from src.features import engineer_features
from src.preprocess import PreprocessingPipeline
from src.train_utils import train_and_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from src.optimize import objective_xgb, objective_lgbm, objective_cat

# Make Optuna quieter during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)

def main():
    print("================================================================")
    print(" NFL DRAFT PREDICTION - ADVANCED STACKING (IMPROVEMENT BRANCH) ")
    print("================================================================")
    
    print("\n[1/7] Loading data...")
    train, test = load_data()
    
    print("\n[2/7] Engineering features...")
    train = engineer_features(train)
    test = engineer_features(test)
    
    X = train.drop(['Drafted', 'Id'], axis=1)
    y = train['Drafted']
    X_test_raw = test.drop(['Id'], axis=1)
    
    cat_cols = ['School', 'Position', 'Player_Type', 'Position_Type', 'Year', 'School_Conference', 'Physical_Archetype']
    
    num_cols_to_clip = [
        'Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 
        'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle',
        'BMI', 'Speed_Score', 'Power_Factor', 'Speed_to_Size_Ratio', 
        'Total_Jump', 'Agility_Score', 'Explosiveness_Index',
        'Bench_to_Weight_Ratio', 'Speed_to_Weight_Efficiency', 
        'Explosiveness_to_Weight_Efficiency', 'Momentum', 'Jump_Power_Index', 'Agility_Speed_Ratio'
    ]
    
    print("\n[3/7] Optimizing XGBoost hyperparameters via Optuna (100 trials)...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y, cat_cols, num_cols_to_clip), n_trials=100)
    best_params_xgb = study_xgb.best_params
    
    print("\n[4/7] Optimizing LightGBM hyperparameters via Optuna (100 trials)...")
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y, cat_cols, num_cols_to_clip), n_trials=100)
    best_params_lgbm = study_lgbm.best_params
    
    print("\n[5/7] Optimizing CatBoost hyperparameters via Optuna (100 trials)...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda trial: objective_cat(trial, X, y, cat_cols, num_cols_to_clip), n_trials=100)
    best_params_cat = study_cat.best_params
    
    models = {
        'xgb': lambda: XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss', tree_method='hist', enable_categorical=True, early_stopping_rounds=50),
        'lgbm': lambda: LGBMClassifier(**best_params_lgbm, verbose=-1, early_stopping_rounds=50),
        'cat': lambda: CatBoostClassifier(**best_params_cat, verbose=0, eval_metric='AUC', cat_features=cat_cols, early_stopping_rounds=50),
        'lr': lambda: LogisticRegression(C=1.0, max_iter=1000)
    }
    
    results = {}
    test_probs = {}
    
    print("\n[6/7] Training Final Models via 5-Fold CV...")
    for name, factory_fn in models.items():
        print(f"\nTraining 5-Fold CV for {name.upper()}...")
        is_linear = (name == 'lr')
        score, oof, fitted_models = train_and_validate(X, y, factory_fn, n_folds=5, cat_cols=cat_cols, num_cols_to_clip=num_cols_to_clip)
        results[name] = {'score': score, 'oof': oof}
        print(f"     {name.upper()} 5-Fold CV ROC AUC: {score:.5f}")
        
        # Predict on test set using fold-averaging (Bagging)
        print(f"     Predicting test set via {len(fitted_models)} fold models...")
        model_test_probs = np.zeros(len(test))
        for pipeline, model in fitted_models:
            X_test_processed = pipeline.transform(X_test_raw)
            model_test_probs += model.predict_proba(X_test_processed)[:, 1] / len(fitted_models)
            
        test_probs[name] = model_test_probs
        
    print("\n[7/7] Meta-Stacking and generating final submission...")
    oof_matrix = np.column_stack([results[m]['oof'] for m in models.keys()])
    test_matrix = np.column_stack([test_probs[m] for m in models.keys()])
    
    # 4. Logistic Regression Stacking Meta-Learner (True Stacking)
    from sklearn.metrics import roc_auc_score
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(oof_matrix, y)
    
    stacked_oof = meta_model.predict_proba(oof_matrix)[:, 1]
    stacked_score = roc_auc_score(y, stacked_oof)
    
    print(f"-> Meta-Model Coefficients (XGB, LGBM, CAT, LR): {np.round(meta_model.coef_[0], 4)}")
    print(f"-> Meta-Model Intercept: {meta_model.intercept_[0]:.4f}")
    
    final_test_probs = meta_model.predict_proba(test_matrix)[:, 1]
        
    submission = pd.DataFrame({
        'Id': test['Id'],
        'Drafted': final_test_probs
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\n" + "="*60)
    print(f" SUCCESS: Advanced predictions saved to submission.csv")
    print(f" Meta-Learner Ensembled CV ROC AUC: {stacked_score:.5f}")
    print("="*60)

if __name__ == "__main__":
    main()