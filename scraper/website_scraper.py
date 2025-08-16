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
import time

# Get a logger instance for this module. The configuration is handled by the main script.
logger = logging.getLogger(__name__)

class WebsiteScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })
        self.visited_urls = set()
        self.scraped_pages = 0

    def is_same_domain(self, url, base_url):
        """Check if URL is from the same domain as base URL"""
        try:
            return urlparse(url).netloc == urlparse(base_url).netloc
        except:
            return False
    
    def normalize_url(self, url):
        """Normalize URL by removing fragments and trailing slashes"""
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized.rstrip('/')

    def extract_page_links(self, soup, base_url):
        """Extract all valid links from the page for crawling"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if not href or href.startswith('#'):
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Only include links from the same domain
            if self.is_same_domain(full_url, base_url):
                normalized_url = self.normalize_url(full_url)
                if normalized_url not in self.visited_urls:
                    links.append(normalized_url)
        
        return links
    
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

    def scrape_website(self, start_urls: list):
        """Main method to scrape website content with depth control"""
        logger.info(f"Starting scrape from {len(start_urls)} seed URL(s).")
        logger.info(f"Max depth: {Config.MAX_DEPTH}, Max pages: {Config.MAX_PAGES}")
        
        self.visited_urls.clear()
        self.scraped_pages = 0
        all_content = []
        all_pdf_urls = set(Config.PDF_URLS) # Start with PDFs from config
        
        try:
            # Initialize crawling with depth tracking
            urls_to_visit = [(url, 0) for url in start_urls]  # (url, depth)
            
            # Crawl pages with depth control
            while urls_to_visit and self.scraped_pages < Config.MAX_PAGES:
                current_url, current_depth = urls_to_visit.pop(0)
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                    
                # Skip if depth limit exceeded
                if current_depth > Config.MAX_DEPTH:
                    logger.info(f"Skipping {current_url} - depth {current_depth} exceeds limit {Config.MAX_DEPTH}")
                    continue
                
                # Mark as visited
                self.visited_urls.add(current_url)
                
                # Scrape the page
                page_content, soup = self.scrape_page_content(current_url)
                if page_content and soup:
                    all_content.append(page_content)
                    self.scraped_pages += 1
                    logger.info(f"✅ Scraped page {self.scraped_pages}/{Config.MAX_PAGES} at depth {current_depth}: {current_url}")

                    # Discover PDFs on this page and add them to the master set
                    pdf_links_on_page = self.find_pdf_links(soup, current_url)
                    all_pdf_urls.update(pdf_links_on_page)
                    
                    # Only extract links if we haven't reached max depth
                    if current_depth < Config.MAX_DEPTH:
                        try:                            
                            # Extract links and add to queue with incremented depth
                            links = self.extract_page_links(soup, current_url)
                            for link in links:
                                if link not in self.visited_urls:
                                    urls_to_visit.append((link, current_depth + 1))
                                    
                            logger.info(f"Found {len(links)} new links at depth {current_depth}")
                            
                        except Exception as e:
                            logger.error(f"Error extracting links from {current_url}: {str(e)}")
                else:
                    logger.warning(f"❌ Failed to scrape or content too short: {current_url}")
                
                # Rate limiting
                time.sleep(1)
                
                # Progress update
                if self.scraped_pages % 10 == 0:
                    logger.info(f"Progress: {self.scraped_pages} pages scraped, {len(urls_to_visit)} URLs in queue")

            logger.info(f"Processing a total of {len(all_pdf_urls)} unique PDF URLs discovered during crawl.")
            
            # --- JSON-First Strategy (keeping your exact logic) ---
            # 1. Process structured data from Syllabus.json
            syllabus_courses = self.load_json_data("Syllabus.json")
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
            logger.info(f"Crawling stats: {self.scraped_pages} pages scraped from {len(self.visited_urls)} visited URLs")
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
            # Remove common non-content elements to get cleaner text
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                script.decompose()

            # --- Improved Title Extraction ---
            title = url.split('/')[-1] # Default title
            if soup.title and soup.title.string:
                title = " ".join(soup.title.string.strip().split())
            elif soup.h1 and soup.h1.string:
                # Fallback to the first H1 tag if title is missing or generic
                title = " ".join(soup.h1.string.strip().split())
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) < Config.MIN_CONTENT_LENGTH:
                return None, None # BUG FIX: Always return a tuple
                
            return ({
                'title': title,
                'url': url,
                'content': text
            }, soup)
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {str(e)}")
            return None, None

    def find_pdf_links(self, soup: BeautifulSoup, url: str):
        """Find all PDF links on the page"""
        try:
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