# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# class LLMIntegration:
#     def __init__(self, model_name: str):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)

#     def query(self, input_text: str) -> str:
#         inputs = self.tokenizer.encode(input_text, return_tensors='pt')
#         outputs = self.model.generate(inputs, max_length=150)
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response

# def load_llm(model_name: str) -> LLMIntegration:
#     return LLMIntegration(model_name)

# def query_llm(prompt, model_name='facebook/bart-large-cnn'):
#     """
#     Queries the specified LLM model with a given prompt and returns the response.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
#     outputs = model.generate(**inputs)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import torch
from vectordb_utils import retrieve_similar_chunks

class LLMIntegration:
    def __init__(self, model_name: str, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)

    def get_embedding(self, text):
        inputs = self.embedding_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings[0].numpy()

    def query(self, question: str, top_k=3) -> str:
        # Step 1: Embed the question
        query_embedding = self.get_embedding(question)
        # Step 2: Retrieve relevant chunks
        relevant_chunks = retrieve_similar_chunks(query_embedding, top_k=top_k)
        context = "\n".join([chunk['text'] for chunk in relevant_chunks])
        # Step 3: Build prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        # Step 4: Query LLM
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model.generate(**inputs, max_length=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def load_llm(model_name: str, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2') -> LLMIntegration:
    return LLMIntegration(model_name, embedding_model_name)