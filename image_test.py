import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
import google.generativeai as genai

from PIL import Image

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

model = genai.GenerativeModel("gemini-pro-vision")

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def image_pipeline(image_file):
    if image_file:
        image_bytes = image_file.getvalue()

        image_parts = [
            {
                "mime_type": image_file.type,
                "data": image_bytes
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("Upload File Please")
    
#streamlit
    
st.set_page_config(page_title="Calorie Counter")

st.header("Gemini Image Calorie Counter")

input = st.text_input("Instructions to the Calorie Calculator", key="input")
image_file = st.file_uploader("Select Image", type=["png","jpg","jpeg"])


if image_file:
    image = Image.open(image_file)
    st.image(image,caption="Here is the image", use_column_width = True)


submit= st.button("Calories")

input_prompt="""
You are an expert in enerfy systems and smart energy devices.\
            you can look at images and give recommendations on what all can be changed int he setting \
        to save more energy. Like making change in the arragement of sitting and lighting in real time rooms \
            or taking enrgy layouts and telling what can be changed to make it better.\
            . Please provide details of every item with its respective recommendation.\
            Also give recommendations if anything has to be replaced or remodelled and try to link its market place \
            
            - Recommendation 1= Please do this as changing this will ensure better 
            - Recommendation 2= Please place the lights at a point like ....
            and so on"""


if submit:
    image_data = image_pipeline(image_file)
    response = get_gemini_response(input_prompt,image_data,input)
    st.subheader("Here is your calorie chart")
    st.write(response)