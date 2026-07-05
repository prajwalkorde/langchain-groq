from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import get_chat_model


def main() -> None:
    llm = get_chat_model()

    prompt = ChatPromptTemplate.from_template(
        "Write a one-sentence definition of {concept}."
    )

    # StrOutputParser converts the model's message object into a plain string.
    # This is useful when the next step expects text instead of a chat message.
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"concept": "retrieval augmented generation"})

    print(type(result))
    print(result)


if __name__ == "__main__":
    main()
