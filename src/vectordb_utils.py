def store_embeddings(embedding, identifier, vector_db):
    # Function to store an embedding in the vector database .
    vector_db[identifier] = embedding

def retrieve_embeddings(identifier, vector_db):
    # Function to retrieve an embedding from the vector database .
    return vector_db.get(identifier, None)

def similarity_search(query_embedding, vector_db, top_k=5):
    # Function to perform a similarity search in the vector database .
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    embeddings = np.array(list(vector_db.values()))
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    
    return [(list(vector_db.keys())[i], similarities[0][i]) for i in top_indices]