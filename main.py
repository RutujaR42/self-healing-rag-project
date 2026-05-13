import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from src.ingestor import ingest_all
from src.vectorstore import build_vectorstore, load_vectorstore
from src.graph import run_pipeline

CHROMA_DIR = "chroma_db"


def setup_knowledge_base():
    """Build the vector store from documents — run this once."""
    print("\n" + "="*50)
    print("SETTING UP KNOWLEDGE BASE")
    print("="*50)
    chunks = ingest_all()
    if not chunks:
        print("No documents found! Add files to data/pdfs, data/texts, or URLs to data/websites.txt")
        return False
    build_vectorstore(chunks)
    print("\nKnowledge base ready!")
    return True


def ask_question(question: str):
    """Run the self-healing RAG pipeline for a question."""
    if not os.path.exists(CHROMA_DIR):
        print("Knowledge base not found! Running setup first...")
        success = setup_knowledge_base()
        if not success:
            return
    return run_pipeline(question)


def interactive_mode():
    """Keep asking questions in a loop until user types 'exit'."""
    print("\n" + "="*50)
    print("SELF-HEALING RAG PIPELINE")
    print("Type your question or 'exit' to quit")
    print("Type 'rebuild' to reload documents")
    print("="*50)

    # Auto setup if chroma_db doesn't exist
    if not os.path.exists(CHROMA_DIR):
        print("\nNo knowledge base found. Building it now...")
        setup_knowledge_base()

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if question.lower() == "rebuild":
            setup_knowledge_base()
            continue

        ask_question(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Healing RAG Pipeline")
    parser.add_argument("--setup",    action="store_true", help="Build knowledge base from documents")
    parser.add_argument("--ask",      type=str,            help="Ask a single question")
    parser.add_argument("--chat",     action="store_true", help="Start interactive chat mode")
    args = parser.parse_args()

    if args.setup:
        setup_knowledge_base()
    elif args.ask:
        ask_question(args.ask)
    elif args.chat:
        interactive_mode()
    else:
        # Default — start interactive mode
        interactive_mode()