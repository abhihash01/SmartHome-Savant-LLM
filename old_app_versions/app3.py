import pandas as pd
from shap_predictor import SHAPPredictor

# Example usage
data_df = pd.read_csv('_lgbm_df.csv')
predictor = SHAPPredictor()
predictions, shap_values = predictor.predict_and_explain('2016-06-10 15:03:00', '2016-06-11 19:41:00', data_df)
print("Predictions:", predictions)
print("SHAP values:", shap_values)
