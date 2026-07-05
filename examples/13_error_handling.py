from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_fixed

from common import get_chat_model


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def ask_with_retry(question: str) -> str:
    """Retry a model call when a temporary error happens."""

    llm = get_chat_model()
    response = llm.invoke([HumanMessage(content=question)])
    return response.content


def main() -> None:
    # Network calls can fail because of rate limits, timeouts, or temporary service issues.
    # Retrying a small number of times makes simple scripts more resilient.
    answer = ask_with_retry("Explain why retries are useful for LLM applications.")
    print(answer)


if __name__ == "__main__":
    main()
