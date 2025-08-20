import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re
import logging
from io import BytesIO
import json
from config import Config
import os

# Get a logger instance for this module. The configuration is handled by the main script.
logger = logging.getLogger(__name__)

class WebsiteScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })

    
    def load_json_data(self, json_file_path):
        """Load structured course data from the Syllabus.json file."""
        try:
            # Construct the absolute path to the JSON file relative to this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one directory to the project root, then to the file
            root_path = os.path.join(base_dir, '..', json_file_path)
            with open(root_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                logger.info(f"Successfully loaded {len(data)} course entries from {json_file_path}")
                return data
        except FileNotFoundError:
            logger.warning(f"Syllabus JSON file not found at {root_path}. Skipping.")
            return []
        except Exception as e:
            logger.error(f"Error loading or parsing {json_file_path}: {str(e)}")
            return []

    def scrape_website(self, url):
        """Main method to scrape website content"""
        logger.info(f"Starting scrape of {url}")
        
        try:
            main_content = self.scrape_page_content(url)
            all_content = [main_content] if main_content else []

        
            # Find PDF links on the page and combine with the static list from config
            found_pdf_links = self.find_pdf_links(url)
            all_pdf_urls = set(found_pdf_links) | set(Config.PDF_URLS)
            logger.info(f"Found {len(found_pdf_links)} PDF links on the page.")
            logger.info(f"Processing a total of {len(all_pdf_urls)} unique PDF URLs (from page and config).")
            
            # --- JSON-First Strategy ---
            # 1. Process structured data from Syllabus.json
            syllabus_courses = self.load_json_data("Syllabus.json")
            # Remplace la boucle for par :
            if syllabus_courses:
                all_content.append({
                    'title': 'Syllabus Courses',
                    'url': 'local_json://Syllabus.json',
                    'content': syllabus_courses  # ← Passe toute la liste, pas chaque cours individuellement
                })

            # 2. Process PDFs, skipping the syllabus PDF to avoid duplication
            for pdf_url in all_pdf_urls:
                # If we used the JSON, skip the corresponding PDF
                if syllabus_courses and "syllabus-book" in pdf_url:
                    logger.info(f"Skipping syllabus PDF '{pdf_url}' because its content was loaded from Syllabus.json.")
                    continue
                pdf_content = self.extract_pdf_text(pdf_url)
                if pdf_content and pdf_content['status'] == 'success':
                    all_content.append({
                        'title': f"PDF: {pdf_url.split('/')[-1]}",
                        'url': pdf_url,
                        'content': pdf_content['text']
                    })
            
            self._log_scraped_data_summary(all_content, all_pdf_urls)
            logger.info(f"Successfully prepared {len(all_content)} documents for processing")
            return all_content
            
        except Exception as e:
            logger.error(f"Error scraping website: {str(e)}")
            return []

    def _log_scraped_data_summary(self, all_content: list, all_pdf_urls: set):
        """Logs a summary of the scraped data before processing."""
        logger.info("--- Scraped Data Summary ---")
        if not all_content:
            logger.info("No data was scraped.")
            logger.info("--------------------------")
            return

        html_docs = []
        pdf_docs = []
        json_docs = []

        for doc in all_content:
            url = doc.get('url', 'Unknown URL')
            if url.startswith('local_json://'):
                json_docs.append(url)
            # Check if it's a PDF by URL or if it was in the collected PDF set
            elif url.lower().endswith('.pdf') or url in all_pdf_urls:
                pdf_docs.append(url)
            else:
                html_docs.append(url)

        logger.info(f"Total documents collected: {len(all_content)}")
        logger.info(f"  - HTML Pages: %s", len(html_docs))
        for url in sorted(html_docs):
            logger.info(f"    - {url}")
        logger.info(f"  - PDF Documents: %s", len(pdf_docs))
        for url in sorted(pdf_docs):
            logger.info(f"    - {url}")
        logger.info(f"  - JSON Sources: %s", len(json_docs))
        for url in sorted(json_docs):
            logger.info(f"    - {url}")
        logger.info("--------------------------")

    def scrape_page_content(self, url):
        """Scrape text content from a web page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # here we remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) < Config.MIN_CONTENT_LENGTH:
                return None
                
            return {
                'title': soup.title.string if soup.title else url.split('/')[-1],
                'url': url,
                'content': text
            }
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {str(e)}")
            return None

    def find_pdf_links(self, url):
        """Find all PDF links on the page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_links = set()
            
            # Look for all links ending with .pdf
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                if href.lower().endswith('.pdf'):
                    pdf_links.add(full_url)
                else:
                    # Also check non-.pdf links if text suggests it's a document
                    text = a.get_text().lower()
                    if 'syllabus' in text or 'pdf' in text or 'document' in text:
                        # Check if this link leads to a PDF
                        try:
                            # Use the already resolved full_url
                            head_response = self.session.head(full_url, timeout=5, allow_redirects=True)
                            content_type = head_response.headers.get('content-type', '').lower()
                            if 'pdf' in content_type:
                               pdf_links.add(full_url)
                        except requests.exceptions.RequestException:
                            pass
            
            return list(pdf_links)  
            
        except Exception as e:
            logger.error(f"Error finding PDF links: {str(e)}")
            return []

    def extract_pdf_text(self, pdf_url):
        """Extract text from PDF with robust error handling"""
        try:
            logger.info(f"Extracting text from PDF: {pdf_url}")
            response = self.session.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Verify PDF content
            content = response.content
            if not content.startswith(b'%PDF-'):
                raise ValueError("Downloaded content is not a valid PDF")
            
            # Extract text using PyMuPDF (fitz)
            with fitz.open(stream=content, filetype="pdf") as doc:
                text = ""
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    if page_text.strip():
                        text += f"Page {page_num + 1}:\n{page_text}\n\n"
                
                # Basic validation
                if not text.strip():
                    raise ValueError("PDF text extraction returned empty content")
                
                logger.info(f"Successfully extracted {len(text)} characters from {len(doc)} pages")
                
                return {
                    'url': pdf_url,
                    'text': text,
                    'page_count': len(doc),
                    'status': 'success'
                }
                
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_url}: {str(e)}")
            return {
                'url': pdf_url,
                'error': str(e),
                'status': 'failed'
            }

    