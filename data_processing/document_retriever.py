from sentence_transformers import SentenceTransformer

from services.qdrant_service import QdrantDatabase

class DocumentRetriever:
    def __init__(self, collection_name: str):
        self.client = QdrantDatabase()
        self.collection_name = collection_name
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 4):
        """Busca documentos no Qdrant com base no texto da query."""
        vector_hits = self.client.search_similar(
            self.collection_name,   
            vector=self.embedder.encode(query),
            limit= top_k)
        
        return [hit.payload for hit in vector_hits]