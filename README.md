#AskMyFile

This project is a Streamlit application that allows users to upload PDF documents, convert them into embeddings, and interact with a local language model (LLM) to retrieve information based on the content of the PDFs.

## Project Structure

```
vectordb-llm-streamlit-app
├── src
│   ├── app.py                  # Main entry point for the Streamlit application
│   ├── pdf_to_embeddings.py     # Functions to convert PDFs into embeddings
│   ├── llm_integration.py       # Integration with the local LLM
│   ├── vectordb_utils.py        # Utility functions for vector database interactions
│   └── ui
│       └── streamlit_ui.py      # User interface components for the Streamlit app
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Files and directories to ignore by Git
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd vectordb-llm-streamlit-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**
   ```
   streamlit run src/app.py
   ```

2. **Upload PDF documents** through the user interface to convert them into embeddings.

3. **Interact with the local LLM** to retrieve information based on the content of the uploaded PDFs.

## Functionality Overview

- **PDF Processing:** The application allows users to upload PDF files, which are then processed to extract text and convert it into embeddings.
- **Local LLM Integration:** The application integrates with a local open-source language model to provide responses based on user queries related to the PDF content.
- **Vector Database:** Embeddings are stored in a vector database, enabling efficient retrieval and similarity searches.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
