import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import fitz  # PyMuPDF
import os
import re

# Load Language Model and Tokenizer
checkpoint = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(checkpoint)
base_model = BartForConditionalGeneration.from_pretrained(checkpoint)

# Function to extract and clean text from PDF
def extract_and_clean_text(pdf_path):
    pdf_text = ""
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pdf_text += page.get_text("text")
    except Exception as e:
        pdf_text = f"Error occurred while extracting text: {str(e)}"
    
    # Clean the extracted text
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", pdf_text)  # Remove special characters
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # Remove extra whitespace
    return cleaned_text

# Create temp directory if it does not exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Streamlit UI
st.title("PDF Text Summarization Tool")

# User Uploads PDF File
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
summarization_mode = st.selectbox("Select Summarization Mode", ["Abstractive", "Extractive"])

if uploaded_file is not None:
    if st.button("Generate Summary"):
        # Save uploaded file temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Extract and clean text from PDF
        cleaned_text = extract_and_clean_text(temp_file_path)

        # Summarize text based on the selected mode
        if summarization_mode == "Abstractive":
            inputs = tokenizer.encode("summarize: " + cleaned_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = base_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:  # Extractive summarization (implement your extractive summarization logic here)
            # Implement extractive summarization logic (e.g., using sentence embeddings, keyword extraction, etc.)
            summary = "Extractive summarization not implemented yet."

        # Display summary
        st.subheader("Summary:")
        st.write(summary)

        # Download Summary as Text File
        if st.button("Download Summary"):
            with open("summary.txt", "w", encoding="utf-8") as summary_file:
                summary_file.write(summary)
            st.success("Summary downloaded successfully as 'summary.txt'")

            # Provide a download link for the user to download the summary file
            st.markdown(get_download_link("summary.txt", "Download Summary Text"), unsafe_allow_html=True)

# Function to generate download link
def get_download_link(file_path, button_text):
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()
    href = f"data:text/plain;charset=utf-8,{file_content}"
    download_link = f'<a href="{href}" download="{file_path}" style="text-decoration:none;">{button_text}</a>'
    return download_link
