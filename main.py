import logging
import os
from dotenv import load_dotenv
import sys

# Add project root to path to allow for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Project Imports ---
from vector_db.faiss_store import VectorStore
from scraper.content_processor import ContentProcessor
from rag.openrouter_rag import OpenRouterRAG
from rag.graph import RAGGraph
from config import config

# --- Setup ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_components():
    """Initializes and returns all the major components of the RAG system."""
    logger.info("Initializing system components...")
    # 0. Pre-flight check for API Key
    if not config.OPENROUTER_API_KEY:
        logger.critical("OPENROUTER_API_KEY is not set. Please create a .env file and add your key.")
        print("\n❌ ERROR: OPENROUTER_API_KEY not found. Please set it in your .env file and restart.")
        return None
    # 1. Initialize the RAG model and test connection
    rag_model = OpenRouterRAG()
    if not rag_model.test_api_connection():
        logger.error("Failed to connect to OpenRouter API. Please check your API key and network connection.")
        return None
    logger.info("OpenRouter API connection successful.")


    # 2. Initialize the content processor (for embeddings)
    processor = ContentProcessor()

    # 3. Initialize the vector store
    vector_store = VectorStore()
    if os.path.exists(f"{config.VECTOR_DB_PATH}.index"):
        logger.info(f"Found existing knowledge base at {config.VECTOR_DB_PATH}. Loading...")
        vector_store.load_index(config.VECTOR_DB_PATH)
        logger.info(f"Vector store loaded with {vector_store.get_stats().get('total_documents', 0)} documents.")
    else :
        logger.error("Knowledge base not found at the specified path.")
        print(f"\n❌ ERROR: Knowledge base not found at '{config.VECTOR_DB_PATH}'.")
        print("Please run 'python build_index.py' first to create the knowledge base.")
        return None
    

    # 4. Initialize the RAG Graph
    rag_graph = RAGGraph(vector_store=vector_store, processor=processor, rag_model=rag_model)
    logger.info("RAG Graph initialized and compiled.")

    return rag_graph

def main():
    """Main function to run the RAG chatbot."""
    rag_graph = initialize_components()
    if not rag_graph:
        return

    print("\n--- 2IS Master's Program RAG Assistant ---")
    print("Ask a question about the program, or type 'exit' to quit.")

    history = []
    try:
        while True:
            question = input("\n> ")
            if question.lower() in ['exit', 'quit']:
                break
            
            if question.lower() == '/reset':
                history = []
                print("\n✨ Conversation history has been cleared.")
                continue
            if question.lower() == '/stats':
                stats = rag_graph.vector_store.get_stats()
                print("\n--- Knowledge Base Stats ---")
                print(f"Status: {stats.get('status', 'N/A')}")
                print(f"Total Documents: {stats.get('total_documents', 'N/A')}")
                print(f"Embedding Dimension: {stats.get('dimension', 'N/A')}")
                continue
            result = rag_graph.run(question,history)
            print(f"\nAssistant: {result['answer']}")
            if result.get('sources'):
                print(f"Sources: {', '.join(result['sources'])}")
            
             # Update history with the latest interaction
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result['answer']})
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Shutting down gracefully.")
if __name__ == "__main__":
    main()
