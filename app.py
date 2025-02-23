import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import PyPDF2
import re
from PIL import Image
import pytesseract
import io
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

def extract_medicines(text):
    # This is a simple example - you might want to use more sophisticated NLP methods
    # or maintain a database of medicine names for better accuracy
    common_medicines = ['aspirin', 'paracetamol', 'ibuprofen']  # expand this list
    found_medicines = []
    
    # Convert text to lowercase for case-insensitive matching
    text = text.lower()
    
    for medicine in common_medicines:
        if medicine in text:
            found_medicines.append(medicine)
    
    return found_medicines

def extract_text_from_image(image):
    try:
        # Convert image to text using pytesseract
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return ""

def analyze_medicines_and_conditions(medicines):
    # Initialize OpenAI
    llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create prompt template for disease analysis
    disease_template = """
    Given the following medicines: {medicines}
    1. What potential conditions or diseases might these medicines be treating?
    2. What are the common uses of these medicines?
    3. List some reliable sources where these medicines can be purchased.
    
    Provide a detailed analysis.
    """
    
    prompt = PromptTemplate(
        input_variables=["medicines"],
        template=disease_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(medicines=", ".join(medicines))
    return response

def main():
    st.title("Smart Prescription Analyzer")
    st.write("Upload a prescription or enter medicine names to get analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF", "Upload Image", "Type Medicine Names"]
    )
    
    medicines = []
    
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                
                st.subheader("Extracted Text:")
                st.text(text_content)
                medicines = extract_medicines(text_content)
                
            except Exception as e:
                st.error(f"Error processing the PDF: {str(e)}")
    
    elif input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Prescription", use_column_width=True)
                text_content = extract_text_from_image(image)
                
                st.subheader("Extracted Text:")
                st.text(text_content)
                medicines = extract_medicines(text_content)
                
            except Exception as e:
                st.error(f"Error processing the image: {str(e)}")
    
    else:  # Type Medicine Names
        medicine_input = st.text_area("Enter medicine names (one per line)")
        if medicine_input:
            medicines = [med.strip().lower() for med in medicine_input.split('\n') if med.strip()]
    
    if medicines:
        st.subheader("Found Medicines:")
        for medicine in medicines:
            st.write(f"- {medicine}")
        
        st.subheader("Analysis:")
        analysis = analyze_medicines_and_conditions(medicines)
        st.write(analysis)
    
    elif input_method != "Type Medicine Names":
        st.warning("No medicines found in the input.")

if __name__ == "__main__":
    main()

