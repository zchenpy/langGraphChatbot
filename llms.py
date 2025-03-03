import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Retrieve OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

##### Set and switch LLM types
# Default to OpenAI (can be changed to "ollama")
DEFAULT_LLM_TYPE = "ollama"

# Model configurations
MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": openai_api_key,
        "model": "gpt-4o-mini"
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "deepseek-r1:14b"
    }
}

def get_model_type() -> str:
    """Return the current default LLM type"""
    return DEFAULT_LLM_TYPE

class LLMInitializationError(Exception):
    """Custom exception for LLM initialization errors"""
    pass

def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> Optional[ChatOpenAI]:
    """Initialize the LLM instance"""
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Available types: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        if llm_type == "ollama":  # ✅ Do not modify the Ollama logic
            return ChatOllama(
                base_url=config["base_url"],
                model=config["model"],
                temperature=0.7
            )
        else:
            return ChatOpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model"],
                temperature=0.7
            )

    except Exception as e:
        logger.error(f"LLM initialization failed: {str(e)}")
        raise LLMInitializationError(f"LLM initialization failed: {str(e)}")

def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> Optional[ChatOpenAI]:
    """Retrieve the LLM instance"""
    try:
        return initialize_llm(llm_type)
    except (LLMInitializationError, ValueError) as e:
        logger.warning(f"LLM initialization failed: {str(e)}. Falling back to default model {DEFAULT_LLM_TYPE}")
        return initialize_llm(DEFAULT_LLM_TYPE)  # Fall back to default model on failure

# Example usage: Only for testing
if __name__ == "__main__":
    try:
        llm_openai = get_llm("openai")
        llm_ollama = get_llm("ollama")

        # This will trigger an error
        llm_invalid = get_llm("invalid_type")  # ❌ Unsupported model

    except (LLMInitializationError, ValueError) as e:  # Catch all initialization errors
        logger.error(f"Program terminated: {str(e)}")
