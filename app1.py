import streamlit as st
from ingest import ingestor
from input_processor import input_processor
from prompt_conversatoin_chain import conversational_chain
import validators
import shutil
import os
from PIL import Image
from image_modelling import image_modelling

faiss_path = 'FAISS_store'
custom_css = """
    <style>
        body {
            background-color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        .sidebar .sidebar-content .block-container {
            margin: 20px;
        }
        .sidebar .sidebar-content .block-container .block-header {
            color: #333333;
        }
        .sidebar .sidebar-content .block-container .block-footer {
            color: #666666;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stButton>button:active {
            background-color: #3e8e41;
        }
    </style>
"""

class app:
    def image_handling(self):
        print("entered image handling")
        st.header("Image Modeler")
        #input = st.text_input("Extra Instructions", key="input")
        image_handling=image_modelling()
        image_file = st.file_uploader("Select Image", type=["png","jpg","jpeg"])
        if image_file:
            image = Image.open(image_file)
            st.image(image,caption="Here is the image", use_column_width = True)
        submit= st.button("Model my image")
        if submit:
            response= image_handling.run(image_file)
            #image_data = image_pipeline(image_file)
            #response = get_gemini_response(input_prompt,image_data,input)
            st.subheader("Here is your response")
            st.write(response)





    def run(self):
        #ingest = ingestor()
        #st.set_page_config("SmartHome Savant")
        st.header("SmartHome Savant Application")

        button_pressed = st.sidebar.radio("Select Option", ["Predictor", "Live Image Modelling", "Knowledge Extractor1"])

        if button_pressed == "Predictor":
            # Add functionality for Option 1 here
            pass
        elif button_pressed == "Live Image Modelling":
            print("radio butoon hit")
            self.image_handling()
        elif button_pressed == "Knowledge Extractor1":
            #toggle_sidebar = st.sidebar.button("Toggle Sidebar")
            #if toggle_sidebar:
                #self.toggle_sidebar_content()
            print("knowledge extractor hit")
            self.toggle_sidebar_content()

    def toggle_sidebar_content(self):
        ingest=ingestor()


        with st.sidebar:
            st.title("Knowledge Bases")
            st.caption("No API required for enterprise models")
            model_temp = st.selectbox("Select Model", ("Gemini", "OpenAI", "LLama", "Mistral","Gemma","Phi"))
            urls = pdf = None
            urls = st.text_input("Enter URLs here")
            pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
            if st.button("Store Knowledge"):
                with st.spinner("Creating Knowledge Base"):
                    if pdfs:
                        try:
                            raw_text = ingest.extract_pdf_text(pdfs)
                            text_chunks = ingest.extract_text_chunks(raw_text)
                            ingest.store_vectors(text_chunks)
                        except Exception as e:
                            st.exception(f"Exception: {e}")
                    if urls:
                        if not validators.url(urls):
                            st.error("Please enter a valid URL")
                        else:
                            raw_text = ingest.extract_url_text([urls])
                            text_chunks = ingest.extract_text_chunks(raw_text)
                            ingest.store_vectors(text_chunks)
                    st.success("Knowledge Base Created")
            st.caption("Don't add both URLs and PDFs together")
            if st.button("Clear Knowledge Base"):
                with st.spinner("Clearing Knowledge Base"):
                    if not os.path.exists(faiss_path):
                        st.write("Knowledge Base Already Clear")
                    else:
                        try:
                            shutil.rmtree(faiss_path)
                        except Exception as e:
                            st.exception(f"Exception: {e}")
                st.success("Knowledge Base Cleared")
        question = st.text_input("Enter Question here")

        submit1 = st.button("Ask LLM")
        submit2 = st.button("Ask Knowledgebase")

        conver_chain = conversational_chain()
        
        inp_processor = input_processor()

        response = []
        if submit1:
            chat = conver_chain.load_chat()
            response = inp_processor.get_response(chat,question)
            st.write("Response: ", response.content)
        if submit2:
            #check if knowledge base exists
            KB_exists = inp_processor.KB_exists()
            if question:
                if KB_exists:
                    chain = conver_chain.load_chain()
                    docs = inp_processor.get_similar_docs(question)
                    response = inp_processor.get_knowledge_response(chain,docs,question)
                    st.write("Response: ", response["output_text"])
                else:
                    st.write("Response: The Knowledge Base Doesn't Exist. Create Knowledge Base to use this feature")
            else:
                st.write("Response: The question field is empty. Enter Question")



if __name__ == "__main__":
    st.markdown(custom_css, unsafe_allow_html=True)
    application = app()
    application.run()