import requests
import json
from typing import List, Dict, Tuple
from config import config
import logging
from base_rag import BaseRAG
logger = logging.getLogger(__name__)

class OpenRouterRAG(BaseRAG):
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.TARGET_URL, # Use your project's URL as the referrer
            "X-Title": "2IS Masters RAG Assistant" # A descriptive title for your app
        }
    
    def test_api_connection(self) -> bool:
        """we will test the OpenRouter API connection."""
        try:
            response = requests.post(
                config.OPENROUTER_API_URL,
                headers=self.headers,
                json={
                    "model": config.OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            error_message = f"API connection test failed: {str(e)}"
            # Provide more specific feedback on common failure reasons
            if e.response is not None:
                if e.response.status_code == 401:
                    error_message = "API connection test failed: 401 Unauthorized. Please check your OPENROUTER_API_KEY in the .env file."
                elif e.response.status_code == 402:
                    error_message = "API connection test failed: 402 Payment Required. Please add credits to your OpenRouter account."
            print(error_message)
            logger.error(error_message)
            return False
        except KeyError as e:
            print(f"Unexpected API response format: {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error during API test: {str(e)}")
            return False
        
    def format_context_with_sources(self, similar_docs: List[Dict]) -> Tuple[str, List[str]] :
        """Format documents into a context string and extract unique source URLs."""
        context = ""
        sources = set()
        seen_content = set()
        for doc in similar_docs:
            if doc['score'] < config.SIMILARITY_THRESHOLD:
                continue
             # Check for duplicate content before adding
            doc_text = doc['document']['content']
            if doc_text in seen_content:
                continue
            seen_content.add(doc_text)

            doc_content = f"Source: {doc['document']['title']} ({doc['document']['url']})\nContent: {doc_text}\n\n"

            # Check if adding the next chunk exceeds the max context length
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
        """Query OpenRouter API with the given prompt and optional context."""
        is_curriculum_question = kwargs.get("is_curriculum_question", False)
        messages = []
        
        # Add conversation history first, if it exists
        if history:
            messages.extend(history)

        if context:
            system_prompt_content = ""
            # Use the explicit flag from the graph state to decide on the prompt
            if is_curriculum_question:
                # Create a special instruction to suggest the PDF download
                final_suggestion = ""
                if config.SYLLABUS_PDF_URL:
                    final_suggestion = f"\n\nAt the end of your response, add this exact sentence: 'For more details, you can download the full syllabus here: {config.SYLLABUS_PDF_URL}'"

                # This is a specialized prompt for syllabus questions to ensure perfect formatting
                system_prompt_content = f"""You are an expert academic advisor for the 2IS Master's program. Your task is to provide a clear, well-structured answer based *only* on the provided course data.

- Format the information logically using markdown headings (e.g., `### Description`) and bullet points for lists.
- Be friendly and helpful in your tone.
- Do not add any information that is not in the context.{final_suggestion}

CONTEXT:
{context}"""
            else:
                # This is the general prompt for all other RAG questions
                system_prompt_content = f"""You are a specialized assistant for the 2IS Master's program. Answer the user's question based *only* on the provided context. Synthesize the information from all provided sources in the context to give a comprehensive and detailed answer. If the context does not contain the information to answer the question, state that you could not find the information in the knowledge base. Do not use any external knowledge.\n\nCONTEXT:\n{context}"""

            messages.append({"role": "system", "content": system_prompt_content})

            
        user_message = prompt
            
        messages.append({"role": "user", "content": user_message})
        try:
            response = requests.post(
                config.OPENROUTER_API_URL, 
                headers=self.headers, 
                json={
                    "model": config.OPENROUTER_MODEL, 
                    "messages": messages, 
                    "temperature": 0.7, 
                    "max_tokens": 8192 
                }, 
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            if 'choices' not in response_data or not response_data.get('choices'):
                error_details = response_data.get('error', response_data)
                logger.error(f"OpenRouter API did not return 'choices'. Response: {json.dumps(error_details, indent=2)}")
                error_message = error_details.get('message', 'The API returned an unexpected response.')
                return f"Error from API: {error_message}"
            return response_data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to OpenRouter API failed: {str(e)}")
            # Provide more specific, user-friendly error messages
            if e.response is not None:
                if e.response.status_code == 401:
                    return "API Error: Unauthorized. Please check your API key."
                if e.response.status_code == 402:
                    return "API Error: Payment Required. Please check your OpenRouter account credits."
            return f"Error: A network problem occurred while connecting to the API."
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from OpenRouter. Response text: {response.text}")
            return "Error: Could not understand the response from the API."
        except Exception as e:
            logger.error(f"An unexpected error occurred in query_openrouter: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
        
    