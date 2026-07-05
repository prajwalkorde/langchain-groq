from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "product_faq.txt"

    documents = TextLoader(str(data_path), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    ).split_documents(documents)

    # Groq provides fast chat models, but not embeddings.
    # For local practice, this project uses a small Hugging Face embedding model.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS stores vectors and can find chunks similar to a user's question.
    vector_store = FAISS.from_documents(chunks, embeddings)

    results = vector_store.similarity_search(
        "What is the best first use case?",
        k=2,
    )

    for result in results:
        print(result.page_content)
        print("---")


if __name__ == "__main__":
    main()
