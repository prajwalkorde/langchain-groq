from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "company_notes.txt"

    # Loaders turn files, web pages, PDFs, or other sources into LangChain Documents.
    loader = TextLoader(str(data_path), encoding="utf-8")
    documents = loader.load()

    # Chunking splits long documents into smaller pieces.
    # Retrieval usually works better with chunks than with huge full documents.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40,
    )
    chunks = splitter.split_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk.page_content)


if __name__ == "__main__":
    main()
