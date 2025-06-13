import streamlit as st
import tempfile
from pdf_to_embeddings import convert_pdf_to_embeddings
from llm_integration import query_llm
from vectordb_utils import store_embeddings, retrieve_embeddings

def render_ui():
    """
    Renders the Streamlit UI for the application.
    """
    st.title("VectorDB LLM Streamlit App")
    st.sidebar.header("Navigation")
    st.sidebar.markdown("Use the sidebar to navigate through the app.")
    st.write("Welcome to the VectorDB LLM Streamlit App!")
    # Add additional UI components as needed

def main():
    st.title("PDF to Embeddings and LLM Interaction")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        try:
            embeddings = convert_pdf_to_embeddings(temp_file_path)  # Pass file path
            store_embeddings(embeddings)
            st.success("PDF processed and embeddings stored successfully!")
        finally:
            # Ensure the temporary file is deleted after processing
            import os
            os.remove(temp_file_path)
        
        user_input = st.text_input("Ask a question to the LLM:")
        
        if st.button("Submit"):
            response = query_llm(user_input)
            st.write("LLM Response:", response)
    
    if st.button("Retrieve Embeddings"):
        embeddings = retrieve_embeddings()
        st.write("Stored Embeddings:", embeddings)

if __name__ == "__main__":
    main()