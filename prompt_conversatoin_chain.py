import os
from dotenv import load_dotenv
from langchain_community.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

class prompt_engineering:
    
    def get_prompt_template(self):
        prompt_template = """ 
        Provide answer to the question from the given context in a detailed format.If the answer \
        is not available in the context, say "answer is not available in the context". Don't make up answers.
        Context: \n {context}? \n
        Question: \n {question} \n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
        return prompt
        
class conversational_chain:
    
    def __init__(self) -> None:
        self.chat_model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.5)

    def load_chat(self):
        return self.chat_model
    
    def load_chain(self):
        prompter = prompt_engineering()
        prompts = prompter.get_prompt_template()
        chain = load_qa_chain(self.chat_model,chain_type="stuff",prompt=prompts)
        
        return chain