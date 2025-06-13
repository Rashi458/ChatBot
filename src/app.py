import streamlit as st
from pdf_to_embeddings import convert_pdf_to_embeddings
from llm_integration import query_llm
from vectordb_utils import store_embeddings, retrieve_embeddings
from ui.streamlit_ui import render_ui

def main():
    st.title("PDF to Vector Database with LLM Integration")
    
    # Render the Streamlit UI
    render_ui()

    # File upload section
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Convert PDF to embeddings
        embeddings = convert_pdf_to_embeddings(uploaded_file)
        
        # Define identifier and vector database 
        identifier = "example_identifier"  # Replace with the actual identifier logic
        vector_db = {}  # Replace with the actual vector database instance
        
        # Store embeddings in vector database
        store_embeddings(embeddings, identifier, vector_db)
        
        st.success("Embeddings stored successfully!")

    # User input for LLM query
    user_input = st.text_input("Ask a question to the LLM:")
    
    if user_input:
        # Query the LLM
        response = query_llm(user_input)
        st.write("LLM Response:", response)

if __name__ == "__main__":
    main()