import pandas as pd
import numpy as np
from scipy.optimize import minimize
from src.data_loader import load_data
from src.features import engineer_features
from src.preprocess import PreprocessingPipeline
from src.train_utils import train_and_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

def main():
    print("================================================================")
    print(" NFL DRAFT PREDICTION - LEAK-FREE ENSEMBLE (2:47 AM VERSION) ")
    print("================================================================")
    
    print("Loading data...")
    train, test = load_data()
    
    print("Engineering features...")
    train = engineer_features(train)
    test = engineer_features(test)
    
    X = train.drop(['Drafted', 'Id'], axis=1)
    y = train['Drafted']
    X_test_raw = test.drop(['Id'], axis=1)
    
    cat_cols = ['School', 'Position', 'Player_Type', 'Position_Type', 'Year', 'School_Conference', 'Physical_Archetype']
    
    # Linear models will use this via impute_and_scale=True, tree models will use target guided clipping bounds
    num_cols_to_clip = [
        'Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 
        'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle',
        'BMI', 'Speed_Score', 'Power_Factor', 'Speed_to_Size_Ratio', 
        'Total_Jump', 'Agility_Score', 'Explosiveness_Index',
        'Bench_to_Weight_Ratio', 'Speed_to_Weight_Efficiency', 
        'Explosiveness_to_Weight_Efficiency'
    ]
    
    models = {
        'xgb': lambda: XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', tree_method='hist', enable_categorical=True, early_stopping_rounds=50),
        'lgbm': lambda: LGBMClassifier(n_estimators=500, verbose=-1, early_stopping_rounds=50),
        'cat': lambda: CatBoostClassifier(n_estimators=500, verbose=0, eval_metric='AUC', cat_features=cat_cols, early_stopping_rounds=50),
        'lr': lambda: LogisticRegression(C=1.0, max_iter=1000)
    }
    
    results = {}
    test_probs = {}
    
    for name, factory_fn in models.items():
        print(f"\nTraining 5-Fold CV for {name.upper()}...")
        # L2-regularized linear models require scaling/OHE
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
        
    print("\n[7/7] Ensembling and generating final submission...")
    oof_matrix = np.column_stack([results[m]['oof'] for m in models.keys()])
    
    # 3. Constrained Weights Optimization for direct ROC AUC maximization
    from sklearn.metrics import roc_auc_score
    def loss_func(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        blend = oof_matrix @ w
        return -roc_auc_score(y, blend)
        
    init_weights = [0.25, 0.25, 0.25, 0.25]
    bounds = [(0, 1)] * 4
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = res.x / np.sum(res.x)
    
    print(f"-> Ensemble weights (XGB, LGBM, CAT, LR): {np.round(opt_weights, 4)}")
    
    final_test_probs = np.zeros(len(test))
    for i, name in enumerate(models.keys()):
        final_test_probs += test_probs[name] * opt_weights[i]
        
    submission = pd.DataFrame({
        'Id': test['Id'],
        'Drafted': final_test_probs
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\n" + "="*60)
    print(f" SUCCESS: Reconstructed predictions saved to submission.csv")
    print(f" Combined Blend Estimated CV ROC AUC: {-res.fun:.5f}")
    print("="*60)

if __name__ == "__main__":
    main()