from typing import List, Dict, Union
import re
from collections import defaultdict
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
            logger.info(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

        # Initialize text splitter only for non-syllabus content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def clean_content(self, text: str) -> str:
        """Basic text cleaning that preserves JSON structure"""
        if not text:
            return ""
        # Don't clean JSON content as it might break the structure
        if text.strip().startswith('{') and text.strip().endswith('}'):
            return text
        # Standard cleaning for regular text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?\-()[\]{}"\'/@]', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s*', r'\1 ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def is_valid_chunk(self, chunk: str) -> bool:
        """Check if a text chunk is valid content"""
        if not chunk or len(chunk.strip()) < 50:
            return False
        # Skip JSON validation as we'll handle it separately
        if chunk.strip().startswith('{'):
            return True
        words = chunk.split()
        if len(words) < 10:
            return False
        junk_patterns = [
            r'^(home|menu|navigation|header|footer)',
            r'^(click here|read more|continue reading)',
            r'^\s*(page \d+|\u00a9|\d+/\d+)',
        ]
        chunk_lower = chunk.lower().strip()
        return not any(re.match(pattern, chunk_lower) for pattern in junk_patterns)

    def process_course_data(self, course: Dict, url: str, doc_idx: int, course_idx: int) -> List[Dict]:
        """Process a single course entry from syllabus data"""
        processed = []
        try:
            # Convert course to JSON string
            course_json = json.dumps(course, ensure_ascii=False)
            
            # Create embedding for the entire course
            embedding = self.embedding_model.encode(course_json)
            embedding = np.array(embedding, dtype='float32')
            
            if embedding.shape[0] != self.embedding_dimension:
                logger.error(f"Invalid embedding dimension for course: {course.get('course_title', 'unknown')}")
                return []

            processed_doc = {
                'title': course.get('course_title', 'Untitled Course'),
                'url': url,
                'content': f"[SYLLABUS_COURSE_DATA] {course_json}",
                'embedding': embedding.tolist(), #c'est notre empriente digitale
                'chunk_id': f"{url}-{course.get('course_title', 'untitled').lower().replace(' ', '-')}-{course_idx}",
                'doc_index': doc_idx,
                'chunk_index': course_idx,  # Each course is one chunk within the syllabus "document"
                'is_syllabus': True,
                'metadata': {
                    'period': course.get('period', ''),
                    'teachers': course.get('teachers', []),
                    'ects': course.get('ects', 0)
                } # pour faciliter le filtrage
            }
            processed.append(processed_doc)
        except Exception as e:
            logger.error(f"Error processing course: {str(e)}")
        return processed

    def process_documents(self, scraped_data: List[Dict]) -> List[Dict]:
        """Process all documents with special handling for syllabus data"""
        if not scraped_data:
            logger.warning("No scraped data provided")
            return []

        processed_docs = []
        syllabus_count = 0

        for doc_idx, doc in enumerate(scraped_data):
            try:
                if not isinstance(doc, dict):
                    continue

                url = doc.get('url', 'Unknown URL')
                content = doc.get('content', '')

                # SPECIAL HANDLING FOR SYLLABUS DATA
                if url.startswith('local_json://') and isinstance(content, list):
                    for course_idx, course in enumerate(content):
                        if not isinstance(course, dict):
                            continue
                        
                        # Validate required fields
                        if not course.get('course_title'):
                            logger.warning(f"Skipping course without title in {url}")
                            continue
                            
                        # Process each course as a complete document
                        course_docs = self.process_course_data(course, url, doc_idx, course_idx)
                        processed_docs.extend(course_docs)
                        syllabus_count += len(course_docs)
                    continue

                # STANDARD PROCESSING FOR REGULAR DOCUMENTS
                if isinstance(content, dict):
                    content = "\n".join(f"{k}: {v}" for k, v in content.items() if v)
                elif not isinstance(content, str):
                    content = str(content)

                clean_text = self.clean_content(content)
                if len(clean_text) < config.MIN_CONTENT_LENGTH:
                    continue

                # Split regular documents into chunks
                chunks = self.text_splitter.split_text(clean_text)
                title = doc.get('title', f'Document {doc_idx}')

                for chunk_idx, chunk in enumerate(chunks):
                    if not self.is_valid_chunk(chunk):
                        continue

                    try:
                        embedding = self.embedding_model.encode(chunk)
                        embedding = np.array(embedding, dtype='float32')
                        
                        processed_doc = {
                            'title': str(title),
                            'url': str(url),
                            'content': chunk,
                            'embedding': embedding.tolist(),
                            'chunk_id': f"{url}-{chunk_idx}",
                            'doc_index': doc_idx,
                            'chunk_index': chunk_idx,
                            'is_syllabus': False
                        }
                        processed_docs.append(processed_doc)
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")

            except Exception as e:
                logger.error(f"Error processing document {doc_idx}: {str(e)}")
                continue

        self._log_processed_data_summary(processed_docs, syllabus_count)
        return processed_docs

    def _log_processed_data_summary(self, processed_docs: List[Dict], syllabus_count: int):
        """Logs a summary of the processed data."""
        logger.info("--- Processed Data Summary ---")
        if not processed_docs:
            logger.info("No data was processed.")
            logger.info("----------------------------")
            return

        chunks_by_url = defaultdict(int)
        for chunk in processed_docs:
            chunks_by_url[chunk['url']] += 1

        logger.info(f"Total chunks created: {len(processed_docs)} (Syllabus Courses: {syllabus_count}, Content Chunks: {len(processed_docs) - syllabus_count})")
        logger.info("Chunks per source URL:")
        for url, count in sorted(chunks_by_url.items()):
            logger.info(f"  - {url}: {count} chunk(s)")
        logger.info("----------------------------")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query into embedding space"""
        try:
            clean_query = self.clean_content(query)
            embedding = self.embedding_model.encode(clean_query)
            return np.array(embedding, dtype='float32')
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            raise