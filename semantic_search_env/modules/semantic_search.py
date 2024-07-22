from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: list):
        return self.model.encode(texts)


class VectorDB:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)

    def add_vectors(self, vectors: np.ndarray):
        if len(vectors.shape) != 2 or vectors.shape[1] != self.index.d:
            raise ValueError(f"Expected vectors of shape (n, {self.index.d})")
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 10):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if len(query_vector.shape) != 2 or query_vector.shape[1] != self.index.d:
            raise ValueError(
                f"Expected query vector of shape (1, {self.index.d})")
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices
