# retrievers/ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrievers.vector_retriever import HybridRetriever


def load_document(file_path: str):
    """Loads a single file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits documents into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def ingest_files(file_paths: list[str], vector_store_path: str = None):
    """
    Full ingestion pipeline: load files, chunk them, build a new hybrid
    retriever index, and save it to disk for future use.
    """
    from config import DATA_DIR
    if vector_store_path is None:
        vector_store_path = str(DATA_DIR / "vector_store")
    all_docs = []
    for path in file_paths:
        raw_docs = load_document(path)
        all_docs.extend(raw_docs)

    chunks = chunk_documents(all_docs)

    retriever = HybridRetriever(documents=chunks, vector_store_path=vector_store_path)
    retriever.save_index()

    return {"files_processed": len(file_paths), "chunks_created": len(chunks)}