from langchain_core.messages import HumanMessage

from common import get_chat_model


def main() -> None:
    llm = get_chat_model()

    # Streaming prints tokens as they arrive.
    # This is useful for chat apps because users see progress immediately.
    for chunk in llm.stream(
        [HumanMessage(content="Give me a short checklist for learning LangChain.")]
    ):
        print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
