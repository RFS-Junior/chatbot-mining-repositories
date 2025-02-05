from langchain_ollama import ChatOllama
from data_processing.document_retriever import DocumentRetriever

class QueryChain:
    def __init__(self, collection_name: str):
        self.document_retriever = DocumentRetriever(collection_name)
        self.llm = ChatOllama(model="gemma2:2b")
        # Inicializa o histórico de conversas como uma lista vazia.
        self.chat_history = []

    def run(self, query: str):
        # 1. Recupera os documentos usando o DocumentRetriever
        docs = self.document_retriever.retrieve(query)
        
        # 2. Gera o prompt incluindo documentos e histórico de conversas
        prompt = self.create_prompt(query, docs, self.chat_history)
        print("Prompt gerado:\n", prompt)
        
        # 3. Passa o prompt para o LLM (Ollama)
        response = self.llm.invoke(prompt)
        
        # 4. Tratamento para extração da resposta
        try:
            # Se o atributo 'content' existir e não for vazio, usamos ele.
            answer = response.content if hasattr(response, 'content') and response.content else None
        except Exception as e:
            print("Erro ao acessar response.content:", e)
            answer = None

        if not answer:
            # Caso não exista response.content ou esteja vazio, utilize uma mensagem padrão ou o str(response)
            answer = str(response) if response else "Não foi possível obter uma resposta adequada."

        # 5. Atualiza o histórico com a interação atual
        self.chat_history.append({
            "user": query,
            "assistant": answer
        })
        
        return answer

    def create_prompt(self, query: str, docs: list, chat_history: list):
        # Constrói o texto do histórico formatando cada turno de conversa.
        history_text = "\n".join([
            f"Usuário: {turn['user']}\nResposta: {turn['assistant']}" 
            for turn in chat_history
        ])
        
        # Concatena o conteúdo dos documentos recuperados.
        doc_text = "\n".join([doc["content"] for doc in docs])
        
        # Monta o prompt final, incluindo seções claras para o histórico, o contexto (documentos) e a consulta atual.
        prompt = f"""
        Você é um assistente especializado em código. Abaixo estão documentos relevantes e o histórico das interações anteriores com o usuário:

        Histórico:
        {history_text}

        INSTRUÇÃO: Por favor, forneça a melhor solução para a pergunta com base no histórico e no contexto apresentados. Se a resposta não for evidente nos documentos, responda com "Não sei".

        Contexto:
        {doc_text}

        Pergunta: {query}

        Resposta:
        """
        
        print(prompt)
        
        return prompt
