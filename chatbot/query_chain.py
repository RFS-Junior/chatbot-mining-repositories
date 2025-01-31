from langchain_ollama import ChatOllama
from data_processing.document_retriever import DocumentRetriever

class QueryChain:
    def __init__(self, collection_name: str):
        self.document_retriever = DocumentRetriever(collection_name)
        self.llm = ChatOllama(model="gemma2:2b")

    def run(self, query: str, top_k: int = 5):
        # 1. Recupera os documentos usando o DocumentRetriever
        docs = self.document_retriever.retrieve(query, top_k)
        
        print(docs)
        
        # 2. Gera o prompt usando os documentos recuperados
        prompt = self.create_prompt(query, docs)
        
        print(prompt)
        
        # 3. Passa o prompt para o LLM (Ollama LLM)
        response = self.llm.invoke(prompt)
        
        return response

    def create_prompt(self, query: str, docs: list):
        """Cria o prompt utilizando os documentos recuperados, de acordo com as diretrizes de engenharia de prompt"""
        # Construa o contexto com os documentos recuperados
        doc_text = "\n".join([doc["content"] for doc in docs])

        prompt = f"""
        Você é um assistente especializado em código. Abaixo estão documentos relevantes para a consulta do usuário:

        INSTRUÇÃO: Por favor, forneça a melhor solução para a pergunta com base no contexto acima. Caso a resposta não seja evidente nos documentos fornecidos, responda com "Não sei".

        Contexto:
        {doc_text}

        Pergunta: {query}

        Resposta:
        """

        return prompt
