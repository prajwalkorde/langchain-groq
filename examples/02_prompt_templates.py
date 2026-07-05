from langchain_core.prompts import ChatPromptTemplate

from common import get_chat_model


def main() -> None:
    llm = get_chat_model()

    # A prompt template lets you reuse the same prompt shape with different inputs.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a practical Python tutor."),
            (
                "human",
                "Teach me {topic} for a {level} learner. Use one tiny code example.",
            ),
        ]
    )

    # The pipe operator creates a chain: prompt output goes into the model.
    chain = prompt | llm

    response = chain.invoke(
        {
            "topic": "list comprehensions",
            "level": "beginner",
        }
    )

    print(response.content)


if __name__ == "__main__":
    main()
