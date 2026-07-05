from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import get_chat_model


def main() -> None:
    llm = get_chat_model()

    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize this business note in one sentence:\n\n{note}"
    )

    action_prompt = ChatPromptTemplate.from_template(
        "Turn this summary into 3 action items:\n\n{summary}"
    )

    # LCEL means LangChain Expression Language.
    # It lets you compose small steps into a readable pipeline.
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    action_chain = action_prompt | llm | StrOutputParser()

    note = """
    Our onboarding dashboard is confusing users. Support tickets mention that
    people cannot find activation metrics. The sales team wants a cleaner
    weekly view before the next customer review.
    """

    summary = summarize_chain.invoke({"note": note})
    actions = action_chain.invoke({"summary": summary})

    print("SUMMARY")
    print(summary)
    print("\nACTIONS")
    print(actions)


if __name__ == "__main__":
    main()
