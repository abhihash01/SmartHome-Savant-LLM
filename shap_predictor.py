import pandas as pd
import joblib
import shap
from shap.explainers import TreeExplainer
import matplotlib.pyplot as plt
import io

class SHAPPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.explainer = TreeExplainer(self.model) 

    def predict_and_explain(self, start_timestamp, end_timestamp, data_df):
        print("input and out put time stamp")
        print(start_timestamp)
        print(end_timestamp)
        filtered_df = data_df[(data_df['time'] >= start_timestamp) & (data_df['time'] <= end_timestamp)]
        cols=['temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure', 
              'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint', 
              'precipProbability', 'year', 'month', 'day', 'weekday', 'weekofyear', 'hour', 'timing']
        print("filtered dataframe")
        print(len(filtered_df))
        filtered_df = filtered_df[cols]
        
        print("Filtered DataFrame:")
        #print(filtered_df)  # Print filtered DataFrame for debugging
        
        # Check if filtered_df is empty
        if filtered_df.empty:
            raise ValueError("Filtered DataFrame is empty.")
        
        # Make predictions using the model
        predictions = self.model.predict(filtered_df)
        
        # Explain the predictions using the SHAP explainer
        shap_values = self.explainer.shap_values(filtered_df)

        shap_values_first_pred = shap_values[0]
        
        # Sort SHAP values for the first prediction
        sorted_indices = shap_values_first_pred.argsort()
        
        # Get top 2 positive and top 2 negative factors
        top_positive_factors = sorted_indices[-2:]
        top_negative_factors = sorted_indices[:2]
        
        # Get feature names and exact values for the top factors
        top_positive_factors_names = filtered_df.columns[top_positive_factors]
        top_positive_factors_values = filtered_df.iloc[0][top_positive_factors]
        
        top_negative_factors_names = filtered_df.columns[top_negative_factors]
        top_negative_factors_values = filtered_df.iloc[0][top_negative_factors]

        top_positive_strings = [f"Top positive feature {i+1}: {name} - {value}" 
                                for i, (name, value) in enumerate(zip(top_positive_factors_names, top_positive_factors_values))]
        top_negative_strings = [f"Top negative feature {i+1}: {name} - {value}" 
                                for i, (name, value) in enumerate(zip(top_negative_factors_names, top_negative_factors_values))]
        

        print("values")
        print(shap_values[0])
        shap.force_plot(self.explainer.expected_value, shap_values[0], filtered_df.iloc[0], show=False)
        img = io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)

        
        # Return both predictions and SHAP plots
        return predictions, img, top_positive_strings, top_negative_strings
