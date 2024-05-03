from flask import Flask, request, jsonify
# General

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


app = Flask(__name__)
with open('preprocessing_pipeline.pkl', 'rb') as file:
    preprocessing_pipeline = pickle.load(file)

model = pickle.load(open("model.pkl", "rb"))



# Load preprocessing and model
@app.route('/api', methods=["GET"])
def predict():
    req_data={}
    
    req_data['gender'] = str(request.args['gender'])
    req_data['age'] = int(request.args['age'])
    req_data['hypertension'] = int(request.args['hypertension'])
    req_data['heart_disease'] = int(request.args['heart_disease'])
    req_data['ever_married'] = str(request.args['ever_married'])
    req_data['work_type'] = str(request.args['work_type'])
    req_data['Residence_type']=str(request.args['Residence_type'])
    req_data['avg_glucose_level'] = float(request.args['avg_glucose_level'])
    req_data['bmi'] = float(request.args['bmi'])
    req_data['smoking_status'] = str(request.args['smoking_status'])
    # Transform the input data using the preprocessing pipeline
    data = pd.DataFrame([req_data])
    transformed_data = preprocessing_pipeline.transform(data)

    # Make predictions using the model
    predictions = model.predict(transformed_data)[0]
    if predictions == 0:
        return jsonify({"output":"not stroke"}), 200
    else:
        return jsonify({"output":"stroke"}), 200

    # Return predictions as JSON response
    

        
if __name__ == '__main__':
    app.run(debug=True)