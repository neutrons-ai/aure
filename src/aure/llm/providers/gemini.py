"""Google Gemini provider."""


def create_llm(config: dict, temperature: float):
    """Create a ``ChatGoogleGenerativeAI`` instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config["api_key"]:
        raise ValueError(
            "LLM_API_KEY or GEMINI_API_KEY must be set for Gemini provider"
        )

    # The google-genai SDK has built-in retries that we can't disable through
    # langchain.  Use invoke_with_timeout() at call sites to cap runaway retries.
    return ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=temperature,
        google_api_key=config["api_key"],
    )
