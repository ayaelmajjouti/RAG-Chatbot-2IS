import google.generativeai as genai
import logging
from typing import List, Dict, Tuple
from config import config
from base_rag import BaseRAG

logger = logging.getLogger(__name__)

class GeminiRAG(BaseRAG):
    """A class to handle interactions with the Google Gemini API."""

    def __init__(self):
        """Initializes the Gemini RAG model and configures the API key."""
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"Gemini model '{config.GEMINI_MODEL}' initialized.")
        except Exception as e:
            logger.critical(f"Failed to configure Gemini API: {e}")
            raise

    def test_api_connection(self) -> bool:
        """Tests the Gemini API connection by generating a small amount of content."""
        try:
            self.model.generate_content("Hello", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
            return True
        except Exception as e:
            error_message = f"API connection test failed: {str(e)}"
            # Provide more specific feedback on common failure reasons
            if "API_KEY_INVALID" in str(e):
                error_message = "API connection test failed: Invalid API Key. Please check your GOOGLE_API_KEY in the .env file."
            print(f"\n❌ {error_message}")
            logger.error(error_message)
            return False

    def format_context_with_sources(self, similar_docs: List[Dict]) -> Tuple[str,List[str]]:
        """Formats documents into a context string and extracts unique source URLs."""
        context = ""
        sources = set()
        seen_content = set()
        for doc in similar_docs:
            if doc['score'] < config.SIMILARITY_THRESHOLD:
                continue
            doc_text = doc['document']['content']
            if doc_text in seen_content:
                continue
            seen_content.add(doc_text)

            doc_content = f"Source: {doc['document']['title']} ({doc['document']['url']})\nContent: {doc_text}\n\n"

            if len(context) + len(doc_content) > config.MAX_CONTEXT_LENGTH:
                break
            context += doc_content
            sources.add(doc['document']['url'])
        return context.strip(), list(sources)

    def format_full_context(self, similar_docs: List[Dict]) -> str:
        """
        Formats all documents into a single context string without a length limit.
        This is used for specialist nodes that need the complete information.
        """
        context = ""
        seen_content = set()
        for doc in similar_docs:
            doc_text = doc['document']['content']
            if doc_text in seen_content:
                continue
            seen_content.add(doc_text)

            # We only need the content for the specialist node's prompt
            doc_content = f"Source Content: {doc_text}\n\n"
            context += doc_content
            
        return context.strip()

    def query(self, prompt: str, context: str = "", history: List[Dict] = None, **kwargs) -> str:
        """
        Queries the Gemini API with the given prompt, context, and history.
        Note: Gemini uses a different message format than OpenAI/OpenRouter.
        """
        # Gemini's chat history format alternates between 'user' and 'model' roles.
        sources = kwargs.get("sources")
        # We will construct the full prompt string here.
        full_prompt = []

        # System prompt for grounding
        if context:
            system_message = f"""You are a specialized assistant for the 2IS Master's program. Answer the user's question based *only* on the provided context. If the context does not contain the information to answer the question, state that you could not find the information in the knowledge base. Do not use any external knowledge.

CONTEXT:
{context}"""
            full_prompt.append(system_message)

        # Simplified history for the prompt
        if history:
            for message in history:
                role = "You" if message['role'] == 'user' else "Assistant"
                full_prompt.append(f"{role}: {message['content']}")

        # The final user message
        user_message = prompt
        if sources:
            user_message += "\n\nRelevant sources: " + ", ".join(sources)
        full_prompt.append(f"You: {user_message}")

        final_prompt = "\n\n".join(full_prompt)

        try:
            response = self.model.generate_content(
                final_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Request to Gemini API failed: {str(e)}")
            if "API_KEY_INVALID" in str(e):
                return "API Error: Invalid API Key. Please check your Google API key."
            return f"Error: An unexpected error occurred while connecting to the Gemini API."
