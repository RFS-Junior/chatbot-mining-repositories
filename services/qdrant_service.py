import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

class QdrantDatabase:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_client()
        return cls._instance
    
    def _initialize_client(self):
        """Initialize Qdrant client with environment configuration"""
        if self._client is None:
            try:
                self._client = QdrantClient(
                    host="localhost",
                    port=6333,
                )
                logging.info("Qdrant client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Qdrant client: {str(e)}")
                raise

    def recreate_collection(self, collection_name: str, vectors_config: VectorParams) -> bool:
        """Create collection if not exists"""
        try:
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            logging.info(f"Collection '{collection_name}' created")
            return True
        except Exception as e:
            logging.warning(f"Collection creation: {str(e)}")
            return False

    def upsert_points(self, collection_name: str, points: List[PointStruct]) -> bool:
        """Batch upsert points with error handling"""
        try:
            self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logging.info(f"Upserted {len(points)} points to '{collection_name}'")
            return True
        except Exception as e:
            logging.error(f"Failed to upsert points: {str(e)}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve collection metadata"""
        try:
            return self._client.get_collection(collection_name).dict()
        except Exception as e:
            logging.error(f"Collection info error: {str(e)}")
            return None

    def search_similar(self, collection_name: str, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic search implementation"""
        try:
            return self._client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True
            )
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return []