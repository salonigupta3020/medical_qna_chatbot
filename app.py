import streamlit as st
import pandas as pd
from chatbot import MedicalChatbot

# Load the dataset
data_path = r'C:\Training\medical_chatbot\MedQuAD\medical_qna_chatbot\medical_qna_chatbot\data\clean_medquaddataset.json'

try:
    # Load dataset using pandas
    data = pd.read_json(data_path)
except FileNotFoundError:
    st.error(f"Error loading dataset: Dataset not found at: {data_path}")
    st.stop()
except ValueError as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Initialize the chatbot
chatbot = MedicalChatbot(data_path)

# Streamlit interface
st.title("Medical Q&A Chatbot")
question = st.text_input("Ask a medical question:")
if st.button("Submit"):
    if question:
        answer = chatbot.get_answer(question)
        st.write(answer)
    else:
        st.error("Please enter a question.")
