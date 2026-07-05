from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys

# Make examples/common.py importable without packaging this practice project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "examples"))

from common import get_chat_model  # noqa: E402


def build_retriever():
    """Load local documents and create a retriever.

    A retriever is the part of a RAG app that finds relevant document chunks for
    a question. Here it is rebuilt at startup to keep the POC simple.
    """

    data_dir = PROJECT_ROOT / "data"
    documents = []

    for path in data_dir.glob("*.txt"):
        loader = TextLoader(str(path), encoding="utf-8")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=60,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store.as_retriever(search_kwargs={"k": 4})


RETRIEVER = build_retriever()


@tool
def search_internal_docs(question: str) -> str:
    """Search internal company and product notes for information."""

    # The agent can call this tool when it needs facts from local documents.
    docs = RETRIEVER.invoke(question)

    if not docs:
        return "No relevant internal notes found."

    return "\n\n".join(doc.page_content for doc in docs)


@tool
def calculate_margin(revenue: float, cost: float) -> float:
    """Calculate profit margin percentage from revenue and cost."""

    profit = revenue - cost
    return round((profit / revenue) * 100, 2)


@tool
def make_implementation_plan(goal: str) -> str:
    """Create a short implementation plan for a product or analytics goal."""

    # This tool is intentionally deterministic.
    # Agents do not need every tool to call another LLM; many useful tools are plain code.
    return (
        f"Implementation plan for {goal}:\n"
        "1. Define the business question and success metric.\n"
        "2. Collect and clean the source documents or data.\n"
        "3. Build a small prototype with 5-10 real user questions.\n"
        "4. Review answer quality with subject-matter experts.\n"
        "5. Add monitoring, feedback capture, and a rollout plan."
    )


def create_agent_executor() -> AgentExecutor:
    llm = get_chat_model(temperature=0.1)
    tools = [search_internal_docs, calculate_margin, make_implementation_plan]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an internal research assistant.

                Rules:
                - Use search_internal_docs when the user asks about company or product facts.
                - Use calculate_margin for financial margin calculations.
                - Use make_implementation_plan when the user asks for a plan.
                - Be concise, practical, and clear.
                - Mention when an answer is based on internal notes.
                """,
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def main() -> None:
    executor = create_agent_executor()

    print("Internal Research Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Ask: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        result = executor.invoke({"input": user_input})
        print("\nAnswer:")
        print(result["output"])
        print()


if __name__ == "__main__":
    main()
