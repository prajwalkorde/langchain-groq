from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from common import get_chat_model


def main() -> None:
    llm = get_chat_model()

    # Basic memory is just message history.
    # In production, you might store this in a database or use LangGraph state.
    history = [
        SystemMessage(content="You are a concise study coach."),
        HumanMessage(content="My goal is to learn LangChain agents."),
        AIMessage(content="Great. We will focus on tools, agent loops, and tracing."),
    ]

    history.append(HumanMessage(content="What should I learn next?"))

    response = llm.invoke(history)

    print(response.content)


if __name__ == "__main__":
    main()
