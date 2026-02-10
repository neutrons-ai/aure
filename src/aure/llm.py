"""
LLM configuration and initialization for the reflectivity analysis agent.

This module provides a unified interface for configuring and accessing LLMs,
supporting OpenAI API, Google Gemini, and local OpenAI-compatible servers 
(like Ollama, LM Studio, or vLLM).

Configuration is done via environment variables:
- LLM_PROVIDER: "openai", "gemini", or "local"
- LLM_API_KEY: API key (may be optional for local servers)
- LLM_MODEL: Model name
- LLM_BASE_URL: Base URL for local servers
- LLM_TEMPERATURE: Temperature setting (default 0.0)
- LLM_TIMEOUT: Timeout in seconds for LLM calls (default 120)

Legacy support:
- OPENAI_API_KEY: Falls back if LLM_API_KEY not set and provider is openai
- GEMINI_API_KEY: Falls back if LLM_API_KEY not set and provider is gemini
"""

import os
import signal
from typing import Optional, Any


class LLMTimeoutError(Exception):
    """Raised when an LLM call times out."""
    pass


def get_llm_timeout() -> int:
    """
    Get the LLM timeout from environment variable.
    
    Set LLM_TIMEOUT in your .env file or environment to control how long
    to wait for LLM responses. Useful for local models that may be slower.
    
    Returns:
        Timeout in seconds (default: 120)
    """
    return int(os.environ.get("LLM_TIMEOUT", "120"))


def _timeout_handler(signum, frame):
    """Signal handler for LLM call timeout."""
    raise LLMTimeoutError("LLM call timed out - possible quota/rate limit issue")


def invoke_with_timeout(llm, prompt, timeout_seconds: int = None) -> Any:
    """
    Invoke an LLM with a timeout to prevent infinite retries.
    
    Args:
        llm: LangChain LLM instance
        prompt: The prompt to send (string or list of messages)
        timeout_seconds: Maximum time to wait (default: LLM_TIMEOUT env var, or 120s)
        
    Returns:
        LLM response
        
    Raises:
        LLMTimeoutError: If the call times out (likely due to retries)
    """
    if timeout_seconds is None:
        timeout_seconds = get_llm_timeout()
    
    # Set up signal-based timeout (Unix only)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        if isinstance(prompt, str):
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
        else:
            response = llm.invoke(prompt)
        return response
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_llm_config() -> dict:
    """
    Get LLM configuration from environment variables.
    
    Returns:
        Dictionary with LLM configuration settings
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    
    # Auto-detect provider from legacy API keys if not explicitly set
    if not provider:
        if os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "openai"  # Default
    
    # Get API key based on provider
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    
    # Default model based on provider
    default_model = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash-lite",
        "local": "llama3",
    }.get(provider, "gpt-4o-mini")
    
    return {
        "provider": provider,
        "api_key": api_key,
        "model": os.environ.get("LLM_MODEL", default_model),
        "base_url": os.environ.get("LLM_BASE_URL"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.0")),
    }


def llm_available() -> bool:
    """
    Check if an LLM is configured and available.
    
    Returns:
        True if LLM can be used, False otherwise
    """
    config = get_llm_config()
    
    # For local providers, we just need a base URL
    if config["provider"] == "local":
        return bool(config["base_url"])
    
    # For cloud providers, we need an API key
    return bool(config["api_key"])


def get_llm(temperature: Optional[float] = None):
    """
    Get a configured LLM instance.
    
    Supports OpenAI API, Google Gemini, and local OpenAI-compatible servers.
    
    Args:
        temperature: Override the default temperature setting
    
    Returns:
        A LangChain chat model instance configured for the selected provider
    
    Raises:
        ValueError: If LLM is not properly configured
    """
    config = get_llm_config()
    temp = temperature if temperature is not None else config["temperature"]
    
    if config["provider"] == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not config["api_key"]:
            raise ValueError(
                "LLM_API_KEY or GEMINI_API_KEY must be set for Gemini provider"
            )
        
        # Note: The google-genai SDK has built-in retries that we can't disable
        # through langchain. We use invoke_with_timeout() to stop runaway retries.
        return ChatGoogleGenerativeAI(
            model=config["model"],
            temperature=temp,
            google_api_key=config["api_key"],
        )
    
    elif config["provider"] == "local":
        from langchain_openai import ChatOpenAI
        
        if not config["base_url"]:
            raise ValueError(
                "LLM_BASE_URL must be set when using local provider. "
                "Example: http://localhost:11434/v1 for Ollama"
            )
        
        # For local servers, api_key might not be required but ChatOpenAI
        # expects one, so we use a placeholder if not provided
        api_key = config["api_key"] or "not-needed"
        
        return ChatOpenAI(
            model=config["model"],
            temperature=temp,
            base_url=config["base_url"],
            api_key=api_key,
            max_retries=0,  # Disable automatic retries
            timeout=float(get_llm_timeout()),
        )
    
    else:  # OpenAI provider (default)
        from langchain_openai import ChatOpenAI
        
        if not config["api_key"]:
            raise ValueError(
                "LLM_API_KEY or OPENAI_API_KEY must be set for OpenAI provider"
            )
        
        return ChatOpenAI(
            model=config["model"],
            temperature=temp,
            api_key=config["api_key"],
            max_retries=0,  # Disable automatic retries
            timeout=float(get_llm_timeout()),
        )


def get_llm_info() -> dict:
    """
    Get information about the current LLM configuration.
    
    Useful for debugging and logging.
    
    Returns:
        Dictionary with LLM info (provider, model, base_url if applicable)
    """
    config = get_llm_config()
    info = {
        "provider": config["provider"],
        "model": config["model"],
        "available": llm_available(),
    }
    
    if config["provider"] == "local":
        info["base_url"] = config["base_url"]
    
    return info
