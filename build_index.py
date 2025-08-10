import logging
import os
import sys
import json
from dotenv import load_dotenv
import argparse

# Add project root to path to allow for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --- Project Imports ---
from scraper.website_scraper import WebsiteScraper
from scraper.content_processor import ContentProcessor
from vector_db.faiss_store import VectorStore
from config import config

# --- Setup ---
load_dotenv()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'build_index.log'))
    ]
)
logger = logging.getLogger(__name__)

def save_debug_json(data, file_path):
    """Helper function to save data to a JSON file for debugging."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Debug data saved to {file_path}")
    except Exception as e:
        logger.error(f"Could not save debug file {file_path}: {e}")

def debug_scraped_documents(scraped_data):
    """Debug function to analyze scraped documents"""
    print("\n" + "="*60)
    print("🔍 DEBUG: ANALYSE DES DOCUMENTS SCRAPÉS")
    print("="*60)
    
    if not scraped_data:
        print("❌ Aucun document scrapé!")
        return
    
    print(f"📊 TOTAL: {len(scraped_data)} documents trouvés\n")
    
    # Grouper par type de source
    sources_stats = {}
    
    for i, doc in enumerate(scraped_data):
        url = doc.get('url', 'Unknown')
        title = doc.get('title', 'No title')
        content = doc.get('content', '')
        
        # Déterminer le type de source
        if url.startswith('local_json://'):
            source_type = "📄 JSON (Syllabus)"
            # Si c'est JSON, content est une liste de cours
            if isinstance(content, list):
                content_preview = f"{len(content)} cours trouvés"
            else:
                content_preview = str(content)[:100]
        elif url.endswith('.pdf'):
            source_type = "📑 PDF"
            content_preview = f"{len(str(content))} caractères"
        else:
            source_type = "🌐 Web Page"
            content_preview = str(content)[:100]
        
        # Stats par type
        if source_type not in sources_stats:
            sources_stats[source_type] = 0
        sources_stats[source_type] += 1
        
        # Afficher les détails
        print(f"[{i+1:2d}] {source_type}")
        print(f"     📍 URL: {url}")
        print(f"     📝 Titre: {title}")
        print(f"     📄 Contenu: {content_preview}{'...' if len(str(content)) > 100 else ''}")
        print()
    
    # Résumé par type
    print("-" * 40)
    print("📈 RÉSUMÉ PAR TYPE:")
    for source_type, count in sources_stats.items():
        print(f"   {source_type}: {count} documents")
    print("="*60)

def build_knowledge_base(force_overwrite=False):
    """Scrapes the target website, processes the content, and builds a FAISS vector store."""
    logger.info(f"Starting to build knowledge base from {len(config.TARGET_URLS)} starting URL(s)...")

    # Define the directory for debug outputs
    DEBUG_DIR = "debug_outputs"

    # Check if an index already exists
    if os.path.exists(f"{config.VECTOR_DB_PATH}.index") and not force_overwrite:
        # If not forced, ask the user interactively
        overwrite = input(f"\n⚠️ An existing knowledge base was found at '{config.VECTOR_DB_PATH}'.\nDo you want to overwrite it? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("\nOperation cancelled by user.")
            return
    
    if os.path.exists(f"{config.VECTOR_DB_PATH}.index"):
        try:
            print("\nOverwriting existing knowledge base...")
            # Safely remove each file if it exists
            index_file = f"{config.VECTOR_DB_PATH}.index"
            docs_file = f"{config.VECTOR_DB_PATH}.docs"
            if os.path.exists(index_file):
                os.remove(index_file)
            if os.path.exists(docs_file):
                os.remove(docs_file)
        except OSError as e:
            logger.warning(f"Could not remove old index files, they will be overwritten. Error: {e}")

    try:
        # 1. Scrape website
        scraper = WebsiteScraper()
        all_scraped_data = []
        for url in config.TARGET_URLS:
            logger.info(f"Scraping content from starting URL: {url}...")
            print(f"Scraping content from starting URL: {url}...")
            all_scraped_data.extend(scraper.scrape_website(url))
        scraped_data = all_scraped_data
        
        if not scraped_data:
            raise ValueError("No content was scraped from the website.")
        
        # Save the raw scraped data for visualization
        save_debug_json(scraped_data, os.path.join(DEBUG_DIR, 'scraped_content_raw.json'))

        # Separate priority data from complementary data for clarity, as per the diagram's logic
        priority_docs = [doc for doc in scraped_data if doc.get('url', '').startswith('local_json://')]
        complementary_docs = [doc for doc in scraped_data if not doc.get('url', '').startswith('local_json://')]

        logger.info(f"Scraping complete. Found {len(priority_docs)} priority document(s) (Syllabus) and {len(complementary_docs)} complementary documents.")
        print(f"✅ Found {len(priority_docs)} priority document(s) (Syllabus) and {len(complementary_docs)} complementary documents.")

        # The debug function can still be called for a more detailed view if needed
        # debug_scraped_documents(scraped_data)
        
        # 2. Process documents and create embeddings
        logger.info("Processing documents and creating embeddings...")
        print("Processing documents and creating embeddings...")
        processor = ContentProcessor()
        processed_docs = processor.process_documents(scraped_data)
        if not processed_docs:
            raise ValueError("No valid documents were processed after chunking.")
        
        # Save the processed data for visualization (without the large embedding vectors)
        docs_for_visualization = [
            {k: v for k, v in doc.items() if k != 'embedding'} for doc in processed_docs
        ]
        save_debug_json(docs_for_visualization, os.path.join(DEBUG_DIR, 'processed_content_for_indexing.json'))
        logger.info(f"Created {len(processed_docs)} text chunks with embeddings.")
        print(f"✅ Created {len(processed_docs)} text chunks with embeddings.")

        # 3. Create and save the vector index
        logger.info("Building and saving the vector search index...")
        print("Building and saving the vector search index...")
        vector_store = VectorStore()
        vector_store.create_index(processed_docs)
        vector_store.save_index(config.VECTOR_DB_PATH)
        logger.info(f"Knowledge base created and saved successfully to '{config.VECTOR_DB_PATH}'!")
        print(f"✅ Knowledge base created and saved to '{config.VECTOR_DB_PATH}'!")
        # Add a final summary block to make results clear
        stats = vector_store.get_stats()
        print("\n--- KNOWLEDGE BASE SUMMARY ---")
        print(f"Total indexed segments: {stats.get('total_documents', 'N/A')}")
        print(f"Embedding dimension: {stats.get('dimension', 'N/A')}")
        print("----------------------------")

    except Exception as e:
        logger.error(f"Fatal error during knowledge base creation: {e}", exc_info=True)
        print(f"\n❌ Failed to create knowledge base. Please check 'build_index.log' for details.")

if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Build or overwrite the RAG knowledge base.")
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite of the existing knowledge base without prompting. Ideal for automated tasks.'
    )
    args = parser.parse_args()

    build_knowledge_base(force_overwrite=args.force)