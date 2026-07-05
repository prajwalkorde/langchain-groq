from langchain_core.messages import HumanMessage, SystemMessage

from common import get_chat_model


def main() -> None:
    llm = get_chat_model()

    # System messages set the assistant's behavior.
    # Human messages contain the user's actual request.
    messages = [
        SystemMessage(content="You explain AI concepts using simple examples."),
        HumanMessage(content="Explain LangChain in 3 bullet points."),
    ]

    response = llm.invoke(messages)

    # LangChain chat models return a message object. The generated text is in .content.
    print(response.content)


if __name__ == "__main__":
    main()
