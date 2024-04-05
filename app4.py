import streamlit as st
from shap_predictor import SHAPPredictor
import pandas as pd
import datetime
import shap
from shap.explainers import TreeExplainer
import joblib
from language_modelling import language_modelling
# Load the data
data_df = pd.read_csv('_lgbm_df.csv')
data_df['time'] = pd.to_datetime(data_df['time'])  # Convert 'time' column to Timestamp objects

# Create SHAPPredictor instance
predictor = SHAPPredictor('lightgbm_model.pkl')

# Streamlit app
st.title('Explainability')
default_start_date = datetime.datetime.strptime('2016/06/10', '%Y/%m/%d').date()

# User input for date range
start_date = st.date_input('Start Date', default_start_date)

end_date = st.date_input('End Date',default_start_date)

# Predict and explain
if st.button('Predict and Explain'):
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Add one day to include the end date
    predictions, shap_plot ,top_positive_strings, top_negative_strings \
= predictor.predict_and_explain(start_timestamp, end_timestamp, data_df)
    
    # Display predictions
    Tot_predictions = predictions.sum()
    #st.write("Predictions:", predictions)
    st.write("Total Predictions:", Tot_predictions)

    #model1 = joblib.load('lightgbm_model.pkl')
    #explainer = TreeExplainer(model1)
    '''
    # Display SHAP plots
    for i, shap_plot in enumerate(shap_plots):
        st.image(shap_plot, caption=f'SHAP Plot for Prediction {i+1}', use_column_width=True)
    '''
    print(top_positive_strings)
    print(top_negative_strings)
    lm= language_modelling()
    response = lm.run(top_negative_strings,top_negative_strings)
    st.write(response)


    shap_plot_bytes = shap_plot.getvalue()
    st.image(shap_plot_bytes, caption=f'Shap plot for first prediction',use_column_width=True)


