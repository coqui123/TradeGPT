"""
LLM API Manager
Handles interactions with the Language Model API
"""
import os
from typing import List
import logging
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

class LLMManager:
    """LLM Manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Load environment variables
        load_dotenv()
        self.endpoint = os.getenv('LLM_API_ENDPOINT')
        self.model_name = os.getenv('LLM_API_MODEL_NAME')
        self.api_token = os.getenv('LLM_API_TOKEN')
        self.client = self.create_llm_client()
        
    def create_llm_client(self) -> ChatCompletionsClient:
        """Create a ChatCompletionsClient instance."""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_token}"
        }
        
        return ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_token),
            headers=headers,
            api_version="2024-02-15-preview"
        )
    
    def get_response(self, messages: List, temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 1000) -> str:
        """Get the response from the ChatCompletionsClient."""
        try:
            response = self.client.complete(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model=self.model_name
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise 