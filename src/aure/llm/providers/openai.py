"""OpenAI provider (default)."""

from ..config import get_llm_timeout


def create_llm(config: dict, temperature: float):
    """Create a ``ChatOpenAI`` instance for the OpenAI API."""
    from langchain_openai import ChatOpenAI

    if not config["api_key"]:
        raise ValueError(
            "LLM_API_KEY or OPENAI_API_KEY must be set for OpenAI provider"
        )

    return ChatOpenAI(
        model=config["model"],
        temperature=temperature,
        api_key=config["api_key"],
        max_retries=0,
        timeout=float(get_llm_timeout()),
    )
