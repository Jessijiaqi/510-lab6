from tempfile import NamedTemporaryFile
import os

import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Feedback on Your Document",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Initialize the chat messages history
if "feedback_messages" not in st.session_state.keys():
    st.session_state.feedback_messages = [
        {"role": "assistant", "content": "Upload your resume or cover letter for feedback."}
    ]

uploaded_file = st.file_uploader("Upload a file")
if uploaded_file:
    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)  # write data from the uploaded file into it
        with st.spinner(
            text="Analyzing your document – hang tight! This should take a moment."
        ):
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                model="gpt-3.5-turbo",
                temperature=0.0,
                system_prompt="You are an expert in providing feedback on resumes and cover letters. Provide detailed, constructive feedback based on the content of the document.",
            )
            index = VectorStoreIndex.from_documents(docs)
    os.remove(tmp.name)  # remove temp file

    if "feedback_engine" not in st.session_state.keys():  # Initialize the feedback engine
        st.session_state.feedback_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=False, llm=llm
        )

# Display a prompt for feedback instead of asking a question
st.write("The document is ready. Click 'Get Feedback' for suggestions on improving your document.")
if st.button("Get Feedback"):
    prompt = "Provide feedback on this resume/cover letter."
    st.session_state.feedback_messages.append({"role": "user", "content": prompt})

for message in st.session_state.feedback_messages:  # Display the prior feedback messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate feedback if the last message was from the user requesting it
if st.session_state.feedback_messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Generating feedback..."):
            response = st.session_state.feedback_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.feedback_messages.append(message)  # Add response to message history
