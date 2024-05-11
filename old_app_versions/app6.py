import streamlit as st
import pandas as pd
import datetime
from PIL import Image
from shap_predictor import SHAPPredictor

# Define custom CSS for styling
custom_css = """
    <style>
        body {
            background-color: #f0f2f6; /* Set background color */
            color: #333333; /* Set text color */
        }
        .sidebar .sidebar-content {
            background-color: #ffffff; /* Set sidebar background color */
        }
        .stButton>button {
            background-color: #4CAF50; /* Set button background color */
            color: white; /* Set button text color */
            padding: 10px 20px;
            border-radius: 4px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049; /* Set button hover background color */
        }
        .stButton>button:active {
            background-color: #3e8e41; /* Set button active background color */
        }
        .stTextInput>div>div>div>input {
            background-color: #ffffff; /* Set text input background color */
            color: #333333; /* Set text input text color */
        }
    </style>
"""

class App:
    def __init__(self) -> None:
        pass
    
    def predictor(self):
        # Load data
        data_df = pd.read_csv('_lgbm_df.csv')
        data_df['time'] = pd.to_datetime(data_df['time'])  # Convert 'time' column to Timestamp objects

        # Create SHAPPredictor instance
        predictor = SHAPPredictor('lightgbm_model.pkl')

        # Streamlit app
        st.title('Predictions')
        default_start_date = datetime.datetime.strptime('2016/06/10', '%Y/%m/%d').date()

        # User input for date range
        start_date = st.date_input('Start Date', default_start_date)
        end_date = st.date_input('End Date', default_start_date)

        # Predict and explain
        if st.button('Predict and Explain'):
            start_timestamp = pd.Timestamp(start_date)
            end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Add one day to include the end date
            predictions, shap_plot, top_positive_strings, top_negative_strings = predictor.predict_and_explain(start_timestamp, end_timestamp, data_df)

            # Display total sum of predictions
            total_prediction_sum = predictions.sum()
            st.write("Total Predictions:", total_prediction_sum)

            # Display SHAP plot for the first prediction
            st.write("SHAP Plot for first prediction:")
            st.image(shap_plot, caption="Shap plot for first prediction", use_column_width=True)

    def run(self):
        st.header("SmartHome Savant Application")
        st.title("")
        button_pressed = st.sidebar.radio("Select Option", ["Predictor", "Live Image Modelling", "Knowledge Extractor1"])
        if button_pressed == "Predictor":
            self.predictor()
        elif button_pressed == "Live Image Modelling":
            # Add functionality for Live Image Modelling
            pass
        elif button_pressed == "Knowledge Extractor1":
            # Add functionality for Knowledge Extractor1
            pass

def main():
    app = App()
    app.run()

if __name__ == "__main__":
    st.markdown(custom_css, unsafe_allow_html=True)
    main()