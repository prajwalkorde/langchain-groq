import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def get_chat_model(temperature: float = 0.2) -> ChatGroq:
    """Create one shared Groq chat model factory for all examples.

    A small helper keeps the example files focused on LangChain concepts instead
    of repeating environment loading code everywhere.
    """

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Copy .env.example to .env and add your Groq API key."
        )

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ChatGroq is LangChain's Groq chat-model wrapper.
    # temperature controls creativity: lower is more focused, higher is more varied.
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )
