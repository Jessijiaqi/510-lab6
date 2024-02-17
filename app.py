import os
import openai
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF

# load environment variables
load_dotenv()

#set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# set page config
st.set_page_config(
    page_title="Document Feedback with GPT",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# define functions
def process_pdf(file):
    # make sure the file is a PDF
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def get_feedback(document_type, document_text):
    # justify the system prompt based on the document type
    if document_type == "resume":
        system_prompt = "You are an expert in resume writing. Provide detailed feedback on this resume."
    elif document_type == "cover_letter":
        system_prompt = "You are an expert in cover letter writing. Provide detailed feedback on this cover letter."
    else:
        return "Document type not supported."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": document_text[:4000]  # limit to 4000 tokens
            }]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error calling OpenAI: {e}"

# main app
uploaded_file = st.file_uploader("Upload a document (resume or cover letter)", type=['pdf'])
document_type = st.selectbox("Document Type", ["Select", "resume", "cover_letter"])

if uploaded_file and document_type != "Select":
    with st.spinner('Processing document...'):
        # process the uploaded PDF
        document_text = process_pdf(uploaded_file)
    
    if document_text:
        feedback = get_feedback(document_type, document_text)
        # display the feedback
        st.write(feedback)
