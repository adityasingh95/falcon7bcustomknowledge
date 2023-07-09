import os
key = input("Please enter OpenAI API Key")
os.environ["OPENAI_API_KEY"] =key

import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil

EMB_OPENAI_ADA = "text-embedding-ada-002"
EMB_INSTRUCTOR_XL = "hkunlp/instructor-xl"
EMB_SBERT_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2" # Chroma takes care if embeddings are None
EMB_SBERT_MINILM = "sentence-transformers/all-MiniLM-L6-v2" # Chroma takes care if embeddings are None

LLM_OPENAI_GPT35 = "gpt-3.5-turbo"
LLM_FLAN_T5_XXL = "google/flan-t5-xxl"
LLM_FLAN_T5_XL = "google/flan-t5-xl"
LLM_FASTCHAT_T5_XL = "lmsys/fastchat-t5-3b-v1.0"
LLM_FLAN_T5_SMALL = "google/flan-t5-small"
LLM_FLAN_T5_BASE = "google/flan-t5-base"
LLM_FLAN_T5_LARGE = "google/flan-t5-large"
LLM_FALCON_SMALL = "tiiuae/falcon-7b-instruct"

# Streamlit app code
st.set_page_config(
    page_title='Q&A Bot for PDF',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)


if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA() ## Intialisation

