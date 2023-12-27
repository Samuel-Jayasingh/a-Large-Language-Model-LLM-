import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import language_tool_python



# Load Language Model and Tokenizer
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# Create LanguageTool instance
tool = language_tool_python.LanguageTool('en-US')

# Streamlit UI
st.title("Text Summarization and Grammar Checking Tool")

# User Input
user_input = st.text_area("Enter text:")

# Summarization Function using BART
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = base_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Button for Summarization
if st.button("Generate Summary"):
    if user_input:
        summary = summarize_text(user_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text for summarization.")

# Grammar Checking Function
def check_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Buttons for Summarization and Gram

if st.button("Check Grammar"):
    if user_input:
        grammar_result = check_grammar(user_input)
        st.subheader("Grammar Check Result:")
        st.write(grammar_result)
    else:
        st.warning("Please enter some text for grammar checking.")
