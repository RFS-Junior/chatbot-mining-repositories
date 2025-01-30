import logging
from data_processing.document_processor import DocumentProcessor
from chatbot.query_chain import QueryChain
import sys

def main():
    # Configuração de logging
    logging.basicConfig(level=logging.INFO)

    # URL do repositório GitHub (substitua pela URL do repositório desejado)
    repository_url = ""

    # Pega o nome da coleção para o Qdrant
    collection_name = repository_url.split("/")[-1]

    # Passo 1: Processar e indexar os documentos do repositório
    logging.info("Iniciando o processamento do repositório...")
    document_processor = DocumentProcessor()

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
    query = "Quantos commits nós temos?"

    try:
        # Resposta usando o modelo de linguagem (LLM) com base nos documentos indexados
        response = query_chain.run(query, top_k=5)
        logging.info(f"Resposta para a consulta '{query}': {response.content}")
    except Exception as e:
        logging.error(f"Erro ao realizar a consulta: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
