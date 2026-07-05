from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import get_chat_model


def format_docs(docs) -> str:
    # RAG prompts need retrieved text inserted as context.
    return "\n\n".join(doc.page_content for doc in docs)


def main() -> None:
    llm = get_chat_model()
    data_path = Path(__file__).resolve().parents[1] / "data" / "product_faq.txt"

    docs = TextLoader(str(data_path), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the context below.
        If the answer is not in the context, say you do not know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    question = "Does InsightPilot replace analysts?"

    # This is the classic RAG flow:
    # 1. Retrieve relevant chunks.
    # 2. Put them into a prompt.
    # 3. Ask the model to answer from that context.
    retrieved_docs = retriever.invoke(question)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": format_docs(retrieved_docs),
            "question": question,
        }
    )

    print(answer)


if __name__ == "__main__":
    main()
