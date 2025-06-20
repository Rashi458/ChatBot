# def store_embeddings(embedding, identifier, vector_db):
#     # Function to store an embedding in the vector database .
#     vector_db[identifier] = embedding

# def retrieve_embeddings(identifier, vector_db):
#     # Function to retrieve an embedding from the vector database .
#     return vector_db.get(identifier, None)

# def similarity_search(query_embedding, vector_db, top_k=5):
#     # Function to perform a similarity search in the vector database .
#     from sklearn.metrics.pairwise import cosine_similarity
#     import numpy as np

#     embeddings = np.array(list(vector_db.values()))
#     similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
#     top_indices = similarities[0].argsort()[-top_k:][::-1]
    
#     return [(list(vector_db.keys())[i], similarities[0][i]) for i in top_indices]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# In-memory vector DB: a list of {'text': ..., 'embedding': ...}
vector_db = []

def store_embeddings(chunk_embeddings):
    """
    Stores a list of {'text', 'embedding'} dicts in the vector DB.
    """
    global vector_db
    vector_db = chunk_embeddings

def retrieve_similar_chunks(query_embedding, top_k=3):
    """
    Retrieves the top_k most similar chunks to the query_embedding.
    Returns a list of dicts with 'text' and 'embedding'.
    """
    if not vector_db:
        return []

    embeddings = np.array([chunk['embedding'] for chunk in vector_db])
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [vector_db[i] for i in top_indices]