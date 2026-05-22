import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class PreprocessingPipeline:
    def __init__(self, cat_cols, num_cols_to_clip=None, impute_and_scale=False):
        self.cat_cols = cat_cols
        self.num_cols_to_clip = num_cols_to_clip
        self.impute_and_scale = impute_and_scale
        self.clipping_bounds = {}
        self.num_cols = []
        self.cat_dtypes = {}
        
        if self.impute_and_scale:
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, X, y=None):
        if self.num_cols_to_clip is not None:
            for col in self.num_cols_to_clip:
                if col in X.columns:
                    q_low = X[col].quantile(0.01)
                    q_high = X[col].quantile(0.99)
                    self.clipping_bounds[col] = (q_low, q_high)
                    
        if not self.impute_and_scale:
            for col in self.cat_cols:
                if col in X.columns:
                    # Learn categories from training data
                    # Use 'missing' for na, and string 'nan' for float nans
                    unique_cats = X[col].astype(str).unique().tolist()
                    if 'missing' not in unique_cats:
                        unique_cats.append('missing')
                    self.cat_dtypes[col] = pd.CategoricalDtype(categories=unique_cats, ordered=False)
        else:
            X_clipped = X.copy()
            for col, bounds in self.clipping_bounds.items():
                if col in X_clipped.columns:
                    q_low, q_high = bounds
                    X_clipped[col] = X_clipped[col].clip(lower=q_low, upper=q_high)
            
            self.num_cols = [c for c in X.columns if c not in self.cat_cols]
            if len(self.num_cols) > 0:
                self.imputer.fit(X_clipped[self.num_cols])
                X_num_imputed = self.imputer.transform(X_clipped[self.num_cols])
                self.scaler.fit(X_num_imputed)
                
            if len(self.cat_cols) > 0:
                X_cat = X_clipped[self.cat_cols].astype(str).fillna('missing')
                self.encoder.fit(X_cat)
                
        return self

    def transform(self, X):
        X = X.copy()
        for col, bounds in self.clipping_bounds.items():
            if col in X.columns:
                q_low, q_high = bounds
                X[col] = X[col].clip(lower=q_low, upper=q_high)
        
        if not self.impute_and_scale:
            for col in self.cat_cols:
                if col in X.columns:
                    # Map unseen categories to 'missing' BEFORE converting to category
                    s = X[col].astype(str)
                    valid_cats = set(self.cat_dtypes[col].categories)
                    s = s.map(lambda x: x if x in valid_cats else 'missing')
                    X[col] = s.astype(self.cat_dtypes[col])
            return X
        else:
            X_num_processed = None
            if len(self.num_cols) > 0:
                X_num_imputed = self.imputer.transform(X[self.num_cols])
                X_num_scaled = self.scaler.transform(X_num_imputed)
                X_num_processed = pd.DataFrame(X_num_scaled, columns=self.num_cols, index=X.index)
                
            X_cat_processed = None
            if len(self.cat_cols) > 0:
                X_cat = X[self.cat_cols].astype(str).fillna('missing')
                X_cat_encoded = self.encoder.transform(X_cat)
                encoded_cols = self.encoder.get_feature_names_out(self.cat_cols)
                X_cat_processed = pd.DataFrame(X_cat_encoded, columns=encoded_cols, index=X.index)
                
            if X_num_processed is not None and X_cat_processed is not None:
                return pd.concat([X_num_processed, X_cat_processed], axis=1)
            elif X_num_processed is not None:
                return X_num_processed
            else:
                return X_cat_processed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)