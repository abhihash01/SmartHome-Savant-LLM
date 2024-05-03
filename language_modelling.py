import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
import google.generativeai as genai

from PIL import Image

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

model = genai.GenerativeModel("gemini-pro")

class language_modelling:
    def get_gemini_response(self,question):
        response=model.generate_content(question)
        return response.text
        
    def run(self,string1,string2):

        input_prompt="""
        You are an expert languge modela and an energy systems expert \
        you are given 2 strings each for positive and negative items and with their respective value\
        First you have to convert it into a very good english form for each of the item like you are explaining the output \
        Next you should give recommendation on how it can be fixed.\
        If its environmental factor or uncontrollable factor, you say it is uncontrollable with the right reasonging
            
            Positive Factors
            - factor 1: Feature name and its value. Full explanation of output and recommendation 
            - factor 2 : Feature name and its value. Full explanation of output and recommendation
            Negative Factors
            - factor 1: Feature name and its value. Full explanation of output and recommendaiton
            - factor 2: Feature name and its value. Full explanation of output and recommendaiotn
            and so on"""

        #input=
        input_prompt=input_prompt+str(string1)+str(string2)
        #print(image_file)
        #image_data = self.image_pipeline(image_file)
        response = self.get_gemini_response(input_prompt)
        return response