from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize Groq LLM
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2  # Low temperature = more factual, less creative
    )


# Prompt template — forces the LLM to only use the provided context
PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question below using ONLY the provided context.
If the context does not contain enough information, say "I don't have enough information to answer this."
Do NOT make up any information.

Context:
{context}

Question:
{question}

Answer:
""")


def format_chunks(chunks: list) -> str:
    """Convert list of Document chunks into a single string."""
    return "\n\n".join([
        f"Source {i+1} ({chunk.metadata.get('source', 'unknown')}):\n{chunk.page_content}"
        for i, chunk in enumerate(chunks)
    ])


def generate_answer(question: str, chunks: list) -> str:
    """Generate an answer from the question and retrieved chunks."""
    print(f"Generating answer for: '{question}'")

    if not chunks:
        return "I don't have enough information to answer this."

    llm = get_llm()
    context = format_chunks(chunks)
    chain = PROMPT | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    answer = response.content
    print(f"Generated answer: {answer[:100]}...")
    return answer