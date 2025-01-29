import logging
from document_processor import DocumentProcessor
from query_chain import QueryChain
import sys
import time

def main():
    # Configuração de logging
    logging.basicConfig(level=logging.INFO)

    # Token do GitHub (substitua pelo seu token real)
    github_token = None

    # URL do repositório GitHub (substitua pela URL do repositório desejado)
    repository_url = None

    # Nome da coleção no Qdrant
    collection_name = "github_repo_x"

    # Passo 1: Processar e indexar os documentos do repositório
    logging.info("Iniciando o processamento do repositório...")
    document_processor = DocumentProcessor(github_token)

    try:
        chunks = document_processor.process_and_index(repository_url)
        logging.info(f"{len(chunks)} chunks processados e indexados no Qdrant.")
    except Exception as e:
        logging.error(f"Erro ao processar o repositório: {str(e)}")
        sys.exit(1)

    # Passo 2: Preparar o modelo de query (QueryChain)
    logging.info("Iniciando o processo de consulta...")
    query_chain = QueryChain(collection_name)

    # Exemplo de consulta
    query = "Qual a data do último commit realizado?"

    try:
        # Resposta usando o modelo de linguagem (LLM) com base nos documentos indexados
        response = query_chain.run(query, top_k=5)
        logging.info(f"Resposta para a consulta '{query}': {response.content}")
    except Exception as e:
        logging.error(f"Erro ao realizar a consulta: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
