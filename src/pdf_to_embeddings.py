from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch

class PDFToEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):  # Updated model name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def extract_text_from_pdf(self, pdf_file):
        """
        Extract text from a PDF file. Accepts either a file path (str) or a file-like object.
        """
        text = ""
        if isinstance(pdf_file, str):  # If it's a file path
            with open(pdf_file, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + " "
        else:  # If it's a file-like object (e.g., UploadedFile)
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text.strip()

    def generate_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def pdf_to_embeddings(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        embeddings = self.generate_embeddings(text)
        return embeddings

def convert_pdf_to_embeddings(pdf_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):  # Updated model name
    """
    Wrapper function to convert a PDF file into embeddings using the PDFToEmbeddings class.
    """
    pdf_to_embeddings = PDFToEmbeddings(model_name=model_name)
    return pdf_to_embeddings.pdf_to_embeddings(pdf_path)