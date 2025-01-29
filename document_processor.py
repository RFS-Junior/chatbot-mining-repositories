import json
import time
from uuid import uuid4
from github import Github
from datetime import datetime
from github.Commit import Commit
from embedder import Embedder
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from pydriller import Repository
import tiktoken

class DocumentProcessor:
    def __init__(self, github_token: str):
        """Inicializa o processador de repositórios com o token do GitHub e o modelo de embeddings."""
        self.github = Github(github_token)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Configuração correta do tokenizer
        self.code_embedder = Embedder("github_repo_x") 
        self.splitter = TokenTextSplitter(
            chunk_size=350,
            chunk_overlap=20
        )
    
    @staticmethod
    def handle_commit(obj):
        """Método estático para serializar objetos Commit do GitHub."""
        if isinstance(obj, Commit):
            return obj.sha
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def rate_limit_check(self):
        """Verifica o limite de requisições da API do GitHub."""
        rate_limit = self.github.get_rate_limit().core
        if rate_limit.remaining < 10:
            reset_time = rate_limit.reset
            sleep_duration = (reset_time - datetime.now()).total_seconds()
            print(f"Token rate limit hit, sleeping for {sleep_duration} seconds")
            time.sleep(max(0, sleep_duration))

    def extract_repository_info(self, repo):
        """Extrai informações principais do repositório."""
        return {
            "url": repo.url,
            "description": repo.description,
            "homepage": repo.homepage,
            "branches": [branch.name for branch in repo.get_branches()],
            "tags": [tag.name for tag in repo.get_tags()],
            "languages": [language for language in repo.get_languages()],
        }

    def extract_commit_info(self, commit):
        """Extrai informações de um commit."""
        return {
            "hash": commit.hash,
            "message": commit.msg,
            "parents": commit.parents,
            "merge": commit.merge,
            "author": {
                "name": commit.author.name,
                "email": commit.author.email,
            },
            "committer": {
                "name": commit.committer.name,
                "email": commit.committer.email,
                "date": commit.committer_date.isoformat() if isinstance(commit.committer_date, datetime) else str(commit.committer_date)
            },
            "modified_files": [],
        }

    def extract_modification_info(self, modification):
        """Extrai informações sobre modificações em arquivos."""
        return {
            "old_path": modification.old_path,
            "new_path": modification.new_path,
            "filename": modification.filename,
            "change_type": modification.change_type.name,
            "added_lines": modification.added_lines,
            "deleted_lines": modification.deleted_lines,
            "nloc": modification.nloc,
            "complexity": modification.complexity,
            "token_count": modification.token_count,
        }

    def extract_issue_info(self, issue):
        """Extrai informações sobre uma issue."""
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "creator": issue.user.login,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
            "labels": [label.name for label in issue.labels],
            "assignees": [assignee.login for assignee in issue.assignees],
            "no_comments": issue.comments,
            "comments": [comment.body for comment in issue.get_comments()],
        }

    def form_metadata(self, repository_url):
        """Forma os metadados coletados do repositório GitHub."""
        repo_name = self.extract_repo_name_from_url(repository_url)
        repo = self.github.get_repo(repo_name)

        self.rate_limit_check()

        repository_data = self.extract_repository_info(repo)
        repository_data["total_commits"] = repo.get_commits().totalCount
        repository_data["total_issues"] = repo.get_issues(state="all").totalCount
        repository_data["total_forks"] = repo.forks_count
        repository_data["total_stars"] = repo.stargazers_count

        repository_data["commits"] = [self.extract_commit_info(commit) for commit in Repository(repository_url).traverse_commits()]
        repository_data["issues"] = [self.extract_issue_info(issue) for issue in repo.get_issues(state="all")]

        repository_data["created_at"] = datetime.utcnow().isoformat()
        repository_data["updated_at"] = datetime.utcnow().isoformat()

        return repository_data

    def extract_repo_name_from_url(self, url):
        """Extrai o nome do repositório a partir da URL fornecida."""
        return url.split("github.com/")[1]
    
    def _chunk_data(self, repository_url: str):
        """Processa os dados extraídos do repositório e divide-os em chunks.""" 
        repository_data = self.form_metadata(repository_url)
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
