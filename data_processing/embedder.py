from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams
from services.qdrant_service import QdrantDatabase

class Embedder:
    def __init__(self, collection_name: str):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantDatabase()
        self.collection_name = collection_name

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )

    def embed_chunks(self, chunks: list):
        """Gera embeddings e indexa no Qdrant"""
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.client.upsert_points(
            collection_name=self.collection_name,
            points=[
                {
                    "id": idx,
                    "vector": emb.tolist(),
                    "payload": chunk
                } for idx, (emb, chunk) in enumerate(zip(embeddings, chunks))
            ]
        )
