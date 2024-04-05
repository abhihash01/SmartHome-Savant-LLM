import pandas as pd
import joblib
import shap
from shap.explainers import TreeExplainer

class SHAPPredictor:
    def __init__(self):
        self.model = joblib.load('lightgbm_model.pkl')
        self.explainer = TreeExplainer(self.model) 

    def predict_and_explain(self, start_timestamp, end_timestamp, data_df):
        filtered_df = data_df[(data_df['time'] >= start_timestamp) & (data_df['time'] <= end_timestamp)]
        cols=['temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure', 
              'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint', 
              'precipProbability', 'year', 'month', 'day', 'weekday', 'weekofyear', 'hour', 'timing']
        filtered_df = filtered_df[cols]
        
        # Make predictions using the model
        predictions = self.model.predict(filtered_df)
        
        # Explain the predictions using the SHAP explainer
        shap_values = self.explainer.shap_values(filtered_df)
        
        # Return both predictions and SHAP values
        return predictions, shap_values