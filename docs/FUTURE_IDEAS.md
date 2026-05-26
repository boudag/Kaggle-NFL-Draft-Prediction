# Future Architecture Ideas

If further optimization is needed beyond PCA Athleticism Embeddings, the following elite Kaggle techniques are available:

### 1. Greedy Hill-Climbing Ensemble (Direct AUC Optimization)
- **The Concept:** Stacking with Logistic Regression optimizes for LogLoss (probability calibration).
- **The Solution:** Write a custom greedy hill-climbing algorithm or rank-averaging technique that searches for percentage weights across base models to explicitly maximize the ROC AUC score.

### 2. Advanced Feature Selection (SHAP / RFECV)
- **The Concept:** High feature counts introduce noise, and `KNNImputer` overfits on noisy features.
- **The Solution:** Implement Recursive Feature Elimination (RFECV) or SHAP value pruning to aggressively drop the bottom 20% of features before models train. This drastically speeds up Optuna and improves generalization.

### 3. TabNet (Deep Learning Base Model)
- **The Concept:** Tree models behave similarly.
- **The Solution:** Introduce a 5th Pillar: **TabNet** by Google, a deep learning architecture explicitly designed for tabular data, providing highly uncorrelated predictions to strengthen the Stacking Meta-Learner.
