import os
from dotenv import load_dotenv

# Try to import streamlit to check for secrets, but handle the case where it's not installed
# or we are not in a streamlit environment.
try:
    import streamlit as st
    _is_streamlit_env = True
except (ImportError, ModuleNotFoundError):
    _is_streamlit_env = False

load_dotenv()

def get_secret(key: str) -> str:
    """
    Retrieves a secret key. It first tries Streamlit's secrets manager,
    then falls back to environment variables. This allows the app to work
    both locally (with a .env file) and when deployed on Streamlit Cloud.
    """
    if _is_streamlit_env and hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

class Config:
    # Target URL for the scraper
    TARGET_URL = "https://miage.ut-capitole.fr/accueil/international/master-innovative-information-systems-2is"
    
    # Add your direct PDF links here. The scraper will download and process them.
    # Example: "https://example.com/path/to/document.pdf"
    PDF_URLS = ["https://miage.ut-capitole.fr/medias/fichier/final-new-2024-flyer-master-2is_1729089997241-pdf?ID_FICHE=573467&INLINE=FALSE","https://miage.ut-capitole.fr/medias/fichier/syllabus-book-2024-25_1738767357979-pdf?ID_FICHE=573467&INLINE=FALSE","https://miage.ut-capitole.fr/medias/fichier/final-en-24-25-livret-accueil-etudiant_1730728450007-pdf?ID_FICHE=573467&INLINE=FALSE"]

     # Automatically find the syllabus PDF URL for providing helpful suggestions
    SYLLABUS_PDF_URL = next((url for url in PDF_URLS if "syllabus-book" in url), None)

    # Scraper settings
    MAX_DEPTH = 3 # NOTE: Not currently implemented. Controls how many 'clicks' away from the start URL to scrape.
    MAX_PAGES = 80 # NOTE: Not currently implemented. A safety limit on the total number of pages to scrape.
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    # Processing settings
    MIN_CONTENT_LENGTH = 100 #Ignore pages with less than 100 characters of text
    CHUNK_SIZE = 1500 # Max characters per text chunk
    CHUNK_OVERLAP = 200  # characters of overlap between chunks to maintain context
    
    # Vector DB settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "vector_db/website_index" #file path to save the knowledge base
    
    # OpenRouter API
    OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "deepseek/deepseek-r1-0528:free" # llm used for generation
    
    # RAG settings
    SIMILARITY_THRESHOLD = 0.3  # Min similarity score for a chunk to be considered relevant
    MAX_CONTEXT_LENGTH = 16000 # Max characters to send to the llm to prevent overly large/expensive API calls

    # Web Search for CRAG (Corrective-RAG)
    TAVILY_API_KEY = get_secret("TAVILY_API_KEY") # Example for Tavily Search API

config = Config()