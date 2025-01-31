import os
import logging
from telegram import Update
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from chatbot.query_chain import QueryChain
from data_processing.document_processor import DocumentProcessor

# Carrega variáveis de ambiente
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

# Configuração de logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inicia a interação e pede o repositório"""
    context.user_data.clear()
    await update.message.reply_text(
        "Olá! 👋 Sou um bot especializado em análise de repositórios GitHub. "
        "Por favor, envie-me o link de um repositório público (ex: https://github.com/hibernate/hibernate-orm)"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manipula todas as mensagens do usuário"""
    user_input = update.message.text
    user_data = context.user_data
    logger = logging.getLogger(__name__)

    try:
        # Verifica se já temos um repositório processado
        if not user_data.get('repo_processed'):
            # Etapa 1: Processar repositório
            if not user_input.startswith('https://github.com/'):
                await update.message.reply_text("Por favor, envie um link válido do GitHub!")
                return

            # Extrai nome do repositório
            repo_url = user_input.strip().rstrip('/')
            repo_name = repo_url.split('/')[-1]
            
            await update.message.reply_text(f"Processando repositório {repo_name}...")

            # Processa e indexa os documentos
            processor = DocumentProcessor(repo_name)
            chunks = processor.process_and_index(repo_url)
            
            # Cria a cadeia de consulta
            user_data['query_chain'] = QueryChain(repo_name)
            user_data['repo_processed'] = True
            user_data['repo_name'] = repo_name
            
            await update.message.reply_text(
                f"Repositório {repo_name} processado com sucesso!\n"
                "Agora você pode fazer perguntas sobre o projeto!"
            )
        else:
            # Etapa 2: Responder perguntas
            query_chain = user_data['query_chain']
            response = query_chain.run(user_input)
            
            # Formata a resposta
            formatted_response = (
                f"**Resposta para '{user_input}':**\n\n"
                f"{response.content}\n\n"
                f"_Repositório: {user_data['repo_name']}_"
            )
            
            await update.message.reply_text(formatted_response)

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        await update.message.reply_text("Ocorreu um erro ao processar sua solicitação. Tente novamente.")

def main():
    """Configura e inicia o bot"""
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Registra handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Inicia o bot
    application.run_polling()
    logging.info("Bot em execução...")

if __name__ == "__main__":
    main()