from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup


def load_pdfs(pdf_dir: str = "data/pdfs") -> list:
    docs = []
    pdf_path = Path(pdf_dir)
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Loading PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load PDF {pdf_file.name}: {e}")
    return docs


def load_texts(text_dir: str = "data/texts") -> list:
    docs = []
    text_path = Path(text_dir)
    for txt_file in text_path.glob("*.txt"):
        print(f"Loading text file: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load text file {txt_file.name}: {e}")
    return docs


def load_websites(websites_file: str = "data/websites.txt") -> list:
    docs = []
    path = Path(websites_file)
    if not path.exists():
        return docs
    urls = [u.strip() for u in path.read_text().splitlines() if u.strip()]
    for url in urls:
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def ingest_all() -> list:
    print("Starting ingestion...")
    all_docs = []
    all_docs.extend(load_pdfs())
    all_docs.extend(load_texts())
    all_docs.extend(load_websites())

    if not all_docs:
        print("No documents found! Add files to data/pdfs, data/texts, or URLs to data/websites.txt")
        return []

    print(f"Total documents loaded: {len(all_docs)}")
    chunks = chunk_documents(all_docs)
    return chunks