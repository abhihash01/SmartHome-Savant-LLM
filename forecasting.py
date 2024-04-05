import pandas as pd
import joblib
import shap
from shap.explainers import TreeExplainer
import numpy as np
import matplotlib.pyplot as plt



# Load the LightGBM model
model = joblib.load('lightgbm_model.pkl')

# Load the SHAP explainer
#explainer = joblib.load('shap_explainer.pkl')
explainer = TreeExplainer(model)
def predict_and_explain(start_timestamp, end_timestamp, data_df, model, explainer):
    filtered_df = data_df[(data_df['time'] >= start_timestamp) & (data_df['time'] <= end_timestamp)]
    cols=['temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure', 
          'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint', 
          'precipProbability', 'year', 'month', 'day', 'weekday', 'weekofyear', 'hour', 'timing']
    filtered_df = filtered_df[cols]
    
    # Make predictions using the model
    predictions = model.predict(filtered_df)
    
    # Explain the predictions using the SHAP explainer
    shap_values = explainer.shap_values(filtered_df)
    
    # Return both predictions and SHAP values
    return predictions, shap_values




# Example usage


# Example usage
data_df = pd.read_csv('_lgbm_df.csv')
predictions, shap_values = predict_and_explain('2016-06-10 15:03:00', '2016-06-11 19:41:00', data_df, model, explainer)

# Print predictions and SHAP values
print("Predictions:", predictions)
print("SHAP values:", shap_values)



def plot_shap_values(predictions, shap_values, data_df):
    # Loop through each prediction
    for i in range(len(predictions)):
        # Create a force plot for the i-th prediction
        shap.force_plot(explainer.expected_value, shap_values[i], data_df.iloc[i], show=False)
        plt.title(f'SHAP Values for Prediction {i+1}, Prediction: {predictions[i]}')
        plt.show()

# Example usage
plot_shap_values(predictions, shap_values, data_df)