########################### Libraries ###########################

# General
import re
import pickle
import joblib


# Data Analysis
import pandas as pd


# Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
# Models

from sklearn.ensemble import GradientBoostingClassifier

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

        
# Load the preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'rb') as file:
    preprocessing_pipeline = pickle.load(file)

model = pickle.load(open("model.pkl", "rb"))


# Create a new instance of the data as a dictionary
new_instance = {
    "gender": 'Female',
    "age": 18.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": 'No',
    "work_type": 'Private',
    "Residence_type": 'Rural',
    "avg_glucose_level": 83.95,
    "bmi": 25.7,
    "smoking_status": 'never smoked'
}

# Convert the new instance to a DataFrame
new_instance_df = pd.DataFrame([new_instance])

# Use the loaded preprocessing pipeline to transform the new instance
transformed_instance = preprocessing_pipeline.transform(new_instance_df)

# Use the model to make predictions
prediction = model.predict(transformed_instance)
print(prediction)