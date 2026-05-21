from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class PreprocessingPipeline:
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.imputer = IterativeImputer(random_state=42, max_iter=10)
        self.encoder = TargetEncoder(cols=cat_cols, smoothing=10)
        self.scaler = StandardScaler()
        self.num_cols = None

    def fit(self, X, y=None):
        # 1. Target Encode
        self.encoder.fit(X, y)
        X_encoded = self.encoder.transform(X)
        
        # 2. Impute
        # Identify numerical columns (everything after encoding should be numerical)
        self.num_cols = X_encoded.columns
        self.imputer.fit(X_encoded)
        X_imputed = self.imputer.transform(X_encoded)
        
        # 3. Scale
        self.scaler.fit(X_imputed)
        return self

    def transform(self, X):
        X_encoded = self.encoder.transform(X)
        X_imputed = self.imputer.transform(X_encoded)
        X_scaled = self.scaler.transform(X_imputed)
        return pd.DataFrame(X_scaled, columns=self.num_cols)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
