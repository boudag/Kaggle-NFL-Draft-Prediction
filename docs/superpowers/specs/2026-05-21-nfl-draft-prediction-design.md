# NFL Draft Prediction - ML Pipeline Design

## Objective
Achieve the highest possible ROC AUC score for predicting whether a player will be drafted.

## Data Components
- **Train Data**: 2781 entries, 16 columns (including target `Drafted`).
- **Test Data**: 15 columns.
- **Target**: `Drafted` (Binary).

## 1. Preprocessing & Data Cleaning
- **Imputation**: Use `IterativeImputer` (BayesianRidge or similar) for missing values in:
  - `Age`, `Sprint_40yd`, `Vertical_Jump`, `Bench_Press_Reps`, `Broad_Jump`, `Agility_3cone`, `Shuttle`.
- **Outlier Treatment**:
  - Analyze distributions for `Height`, `Weight`, `Broad_Jump`.
  - Apply Clipping (Winsorization) to handle extreme outliers while preserving data.
- **Scaling**: `StandardScaler` for all numerical features to assist models that aren't scale-invariant (though GBDTs are, scaling helps with some imputation/encoding steps).

## 2. Advanced Feature Engineering
- **Required Features**:
  - `Power_Factor` = `Weight` * `Vertical_Jump`
  - `Speed_to_Size_Ratio` = `Sprint_40yd` / `Weight`
- **Additional Features**:
  - `Total_Jump` = `Vertical_Jump` + `Broad_Jump`
  - `Agility_Score` = `Agility_3cone` + `Shuttle`
  - `BMI` = `Weight` / (`Height` / 100)**2 (If height is in cm)
- **Year Handling**: Treat `Year` as a categorical feature to capture cohort-specific drafting trends.

## 3. Encoding Strategy
- **High Cardinality**: `School`, `Position`.
  - Use `TargetEncoder` with smoothing to capture the mean of the target per category without overfitting.
- **Low Cardinality**: `Player_Type`, `Position_Type`, `Year`.
  - Use `TargetEncoder` or `OneHotEncoder`.
- **Validation-friendly Encoding**: Ensure encoding is performed within each CV fold to prevent data leakage.

## 4. Model Architecture & Ensembling
- **Base Models**:
  - **XGBoost**: Robust to variety of data, handles missing values internally (though we impute).
  - **LightGBM**: Fast, accurate on large/sparse data.
  - **CatBoost**: Excellent with categorical features natively, though we'll provide encoded ones too.
- **Cross-Validation**: 10-Fold Stratified CV.
- **Ensembling Strategy**:
  - Final prediction will be a weighted average of the probabilities from the three models.
  - Weights will be determined based on CV performance.

## 5. Automated Optimization
- **Hyperparameter Tuning**: `Optuna` to optimize `roc_auc`.
- **Feature Selection**:
  - Monitor feature importance.
  - Prune features with zero or near-zero importance across models.
- **Iterative Improvement**: Check for 0.001 ROC AUC improvement per run.

## 6. Success Criteria
- Maximize ROC AUC on a held-out test set or CV.
- Robust performance across different folds.
