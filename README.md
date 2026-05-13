# Self-Healing RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that critiques its own output 
and retries automatically when hallucinations are detected.

## How It Works

1. **Ingest** — Load PDFs, text files, and websites into chunks
2. **Retrieve** — Find relevant chunks from ChromaDB vector store
3. **Generate** — Llama 3.1 writes an answer from retrieved chunks
4. **Critic** — A second LLM checks if the answer is grounded in sources
5. **Retry** — If hallucination detected, reformulate query and try again
6. **Fallback** — After 2 retries, gracefully respond "I don't have enough info"

## Tech Stack

- **LangGraph** — Stateful retry pipeline
- **LangChain** — LLM orchestration
- **Groq + Llama 3.1** — Free, fast LLM inference
- **ChromaDB** — Local vector store
- **sentence-transformers** — Local embeddings (all-MiniLM-L6-v2)
- **Python 3.10+**

##  Project Structure
├── src/
│   ├── ingestor.py      # Load PDFs, texts, websites → chunks
│   ├── vectorstore.py   # ChromaDB store and retrieval
│   ├── generator.py     # LLM answer generation
│   ├── critic.py        # Hallucination detection agent
│   └── graph.py         # LangGraph self-healing pipeline
├── data/
│   ├── pdfs/            # Add your PDF files here
│   ├── texts/           # Add your text files here
│   └── websites.txt     # Add URLs to scrape here
├── main.py              # Entry point
└── requirements.txt

-------------------------------------------------------------

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/RutujaR42/self-healing-rag-project.git
cd self-healing-rag-project
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file:

GROQ_API_KEY=your-groq-api-key-here

Get a free key at [console.groq.com](https://console.groq.com)

### 5. Add your documents
- Drop PDFs into `data/pdfs/`
- Drop text files into `data/texts/`
- Add URLs to `data/websites.txt`

### 6. Run the pipeline
```bash
# Interactive chat mode
python main.py --chat

# Ask a single question
python main.py --ask "What is machine learning?"

# Rebuild knowledge base
python main.py --setup
```

## Example Output

==================================================
Question: What is machine learning?
[RETRIEVE] Searching for: 'What is machine learning?'
[GENERATE] Generating answer...
[CRITIC] Evaluating answer...
Critic verdict: PASS
[DECISION] Answer approved! 
FINAL ANSWER: Machine learning is a subset of AI...
VERDICT: PASS
RETRIES: 0
