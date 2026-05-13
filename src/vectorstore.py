import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Directory where ChromaDB will save data locally
CHROMA_DIR = "chroma_db"

# Embedding model — runs locally, no API needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """Load the embedding model."""
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks: list) -> Chroma:
    """Convert chunks into vectors and store in ChromaDB."""
    print(f"Building vector store with {len(chunks)} chunks...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Vector store built and saved!")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB from disk."""
    print("Loading existing vector store...")
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def retrieve_chunks(query: str, vectorstore: Chroma, k: int = 3) -> list:
    """Find top-k most relevant chunks for a query."""
    print(f"Retrieving top {k} chunks for query: '{query}'")
    results = vectorstore.similarity_search(query, k=k)
    return results