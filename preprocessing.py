from sklearn.base import BaseEstimator, TransformerMixin
class HandleSmokingStatus(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy["smoking_status"].fillna(value="Unknown", inplace=True)
        X_copy["smoking_not_found"] = (X_copy["smoking_status"] == "Unknown").astype(int)
        return X_copy