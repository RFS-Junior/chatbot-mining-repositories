import json
from uuid import uuid4
from data_processing.embedder import Embedder
from langchain.text_splitter import TokenTextSplitter
import tiktoken

from services.github_service import GitHubService

class DocumentProcessor:
    def __init__(self):
        """Inicializa o processador de repositórios com o token do GitHub e o modelo de embeddings."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.code_embedder = Embedder("github_repo_x") 
        self.splitter = TokenTextSplitter(
            chunk_size=350,
            chunk_overlap=20
        )
        self.github = GitHubService()
    
    def _chunk_data(self, repository_url: str):
        """Processa os dados extraídos do repositório e divide-os em chunks.""" 
        repository_data = self.github.form_metadata(repository_url)
        json_str = json.dumps(repository_data, indent=1, default=self.handle_commit)
        chunks = self.splitter.split_text(json_str)

        return [{
            "id": str(uuid4()),
            "content": chunk,
            "tokens": len(self.tokenizer.encode(chunk)),
            "metadata": {
                "source": repository_data["url"],
                "chunk_type": "repository_data"
            }
        } for chunk in chunks]

    def process_and_index(self, repository_url: str):
        """Processa os dados do repositório e os indexa no Qdrant."""
        chunks = self._chunk_data(repository_url)
        self.code_embedder.embed_chunks(chunks)
        return chunks