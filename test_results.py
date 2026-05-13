from dotenv import load_dotenv
load_dotenv()

from src.ingestor import ingest_all
from src.vectorstore import build_vectorstore
from src.graph import run_pipeline

# Build vector store first (only need to do this once)
chunks = ingest_all()
build_vectorstore(chunks)

# Now run the full self-healing pipeline!
run_pipeline("What is machine learning?")