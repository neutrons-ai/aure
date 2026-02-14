"""Local / OpenAI-compatible provider (Ollama, LM Studio, vLLM, etc.)."""

from ..config import get_llm_timeout


def create_llm(config: dict, temperature: float):
    """Create a ``ChatOpenAI`` instance pointed at a local server."""
    from langchain_openai import ChatOpenAI

    if not config["base_url"]:
        raise ValueError(
            "LLM_BASE_URL must be set when using the local provider. "
            "Example: http://localhost:11434/v1 for Ollama"
        )

    # Local servers may not require an API key, but ChatOpenAI expects one.
    api_key = config["api_key"] or "not-needed"

    return ChatOpenAI(
        model=config["model"],
        temperature=temperature,
        base_url=config["base_url"],
        api_key=api_key,
        max_retries=0,
        timeout=float(get_llm_timeout()),
    )
