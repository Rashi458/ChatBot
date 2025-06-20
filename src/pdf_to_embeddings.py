from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch

class PDFToEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def extract_text_chunks_from_pdf(self, pdf_file):
        """
        Extract text chunks from a PDF file. Here, each page is a chunk.
        Returns a list of strings (one per page).
        """
        chunks = []
        if isinstance(pdf_file, str):
            with open(pdf_file, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        chunks.append(page_text.strip())
        else:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    chunks.append(page_text.strip())
        return chunks

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings[0].numpy()

    def pdf_to_embeddings(self, pdf_path):
        chunks = self.extract_text_chunks_from_pdf(pdf_path)
        chunk_embeddings = []
        for chunk in chunks:
            embedding = self.generate_embedding(chunk)
            chunk_embeddings.append({'text': chunk, 'embedding': embedding})
        return chunk_embeddings

def convert_pdf_to_embeddings(pdf_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Wrapper function to convert a PDF file into a list of {'text', 'embedding'} dicts.
    """
    pdf_to_embeddings = PDFToEmbeddings(model_name=model_name)
    return pdf_to_embeddings.pdf_to_embeddings(pdf_path)