from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseRAG(ABC):
    """
    An abstract base class that defines the common interface for all RAG model handlers.
    This ensures that any new RAG provider (like Gemini, OpenAI, etc.) will have a consistent
    set of methods that the main graph can rely on.
    """

    @abstractmethod
    def test_api_connection(self) -> bool:
        """Tests the API connection and credentials."""
        pass

    @abstractmethod
    def format_context_with_sources(self, similar_docs: List[Dict]) -> Tuple[str, List[str]]:
        """Formats documents into a context string and extracts source URLs."""
        pass

    @abstractmethod
    def query(self, prompt: str, context: str = "", history: List[Dict] = None, **kwargs) -> str:
        """Queries the underlying LLM with a prompt, context, and history."""
        pass

