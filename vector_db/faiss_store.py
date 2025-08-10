import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Optional
from config import config
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2

    def create_index(self, documents: List[Dict]):
        """Updated to handle syllabus documents"""
        if not documents:
            raise ValueError("No documents provided")

        logger.info(f"Creating FAISS index with {len(documents)} documents")

        embeddings = []
        self.documents = []
        
        for i, doc in enumerate(documents):
            try:
                if not isinstance(doc, dict) or 'embedding' not in doc:
                    continue
                    
                embedding = np.array(doc['embedding'], dtype='float32')
                if embedding.shape[0] != self.dimension:
                    continue
                    
                # Preserve all metadata including syllabus flag
                clean_doc = {
                    'title': str(doc.get('title', '')),
                    'url': str(doc.get('url', '')),
                    'content': str(doc.get('content', '')),
                    'chunk_id': str(doc.get('chunk_id', f'chunk_{i}')),
                    'doc_index': doc.get('doc_index', 0),
                    'chunk_index': doc.get('chunk_index', 0),
                    'is_syllabus': doc.get('is_syllabus', False),
                    'metadata': doc.get('metadata', {})
                }
                
                embeddings.append(embedding)
                self.documents.append(clean_doc)
                
            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")
                continue

        if not embeddings:
            raise ValueError("No valid embeddings found")

        embeddings_array = np.vstack(embeddings)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
        
        syllabus_count = sum(1 for d in self.documents if d.get('is_syllabus'))
        logger.info(f"Created index with {len(self.documents)} docs ({syllabus_count} syllabus)")

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3, 
                     filter_syllabus: bool = False) -> List[Dict]:
        """Enhanced search with syllabus filtering"""
        if self.index is None:
            raise ValueError("Index not initialized")

        try:
            query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
            if query_embedding.shape[1] != self.dimension:
                raise ValueError("Dimension mismatch")

            # Search more documents if filtering will be applied
            search_k = min(k * 3, len(self.documents)) if filter_syllabus else k
            distances, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                    
                doc = self.documents[idx]
                # Apply syllabus filter if requested
                if filter_syllabus and not doc.get('is_syllabus', False):
                    continue
                    
                results.append({
                    'document': doc,
                    'score': 1.0 / (1.0 + float(distances[0][i])),
                    'distance': float(distances[0][i])
                })
                
            return sorted(results, key=lambda x: x['score'], reverse=True)[:k]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    def find_syllabus_courses(self, query_embedding: np.ndarray, 
                         year: str = None, k: int = 10) -> List[Dict]:
        """Specialized syllabus course search"""
        try:
            # First get all potential syllabus matches
            candidates = self.similarity_search(
                query_embedding,
                k=min(k*2, len(self.documents)),
                filter_syllabus=True
            )
            
            # Filter by year if specified
            if year:
                year = str(year).lower()
                filtered = []
                for result in candidates:
                    period = result['document'].get('metadata', {}).get('period', '').lower()
                    if any(term in period for term in [year, f"y{year}", f"m{year}"]):
                        filtered.append(result)
                return filtered[:k]
                
            return candidates[:k]
            
        except Exception as e:
            logger.error(f"Syllabus search failed: {str(e)}")
            return []
    def save_index(self, path: str = None):
        """Save index with metadata support"""
        if path is None:
            path = config.VECTOR_DB_PATH
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            faiss.write_index(self.index, f"{path}.index")
            
            with open(f"{path}.docs", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'dimension': self.dimension,
                    # Add version info for compatibility
                    'version': '1.1'  
                }, f)
            
            logger.info(f"Saved index with {len(self.documents)} documents")

            
            
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise


    def load_index(self, path: str = None):
        """Load index from disk with validation"""
        if path is None:
            path = config.VECTOR_DB_PATH
            
        index_file = f"{path}.index"
        docs_file = f"{path}.docs"
        
        # Check if files exist and are not empty
        if not os.path.exists(index_file) or os.path.getsize(index_file) == 0:
            raise FileNotFoundError(f"FAISS index file missing or empty: {index_file}")
        
        if not os.path.exists(docs_file) or os.path.getsize(docs_file) == 0:
            raise FileNotFoundError(f"Documents file missing or empty: {docs_file}")

        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load documents
            with open(docs_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.dimension = data['dimension']
            
            # Validate loaded data
            if len(self.documents) != self.index.ntotal:
                raise ValueError(f"Mismatch between number of documents ({len(self.documents)}) and index size ({self.index.ntotal})")
            
            logger.info(f"Successfully loaded index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up any corrupted files"""
        try:
            path = config.VECTOR_DB_PATH
            for ext in ['.index', '.docs']:
                file_path = f"{path}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed corrupted file: {file_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_stats(self):
        """Enhanced statistics with syllabus info"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        syllabus_count = sum(1 for d in self.documents if d.get('is_syllabus'))
        
        return {
            "status": "initialized",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "syllabus_documents": syllabus_count,
            "regular_documents": len(self.documents) - syllabus_count
        }