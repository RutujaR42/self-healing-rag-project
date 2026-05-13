import os
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Load Groq LLM."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def critic_agent(question: str, answer: str, chunks: list) -> dict:
    """
    Evaluate if the answer is grounded in the retrieved chunks.
    Returns a dict with 'verdict' (PASS/FAIL) and 'reason'.
    """
    llm = get_llm()

    # Build context from chunks
    context = "\n\n".join([c.page_content for c in chunks])

    prompt = f"""You are a strict fact-checker. Your job is to verify if the answer is grounded in the provided context.

QUESTION: {question}

CONTEXT (retrieved documents):
{context}

ANSWER TO EVALUATE:
{answer}

Instructions:
- Reply PASS if the answer is fully supported by the context above.
- Reply FAIL if the answer contains information NOT found in the context (hallucination).
- Reply FAIL if the answer is vague or says "I don't know" when context has the answer.
- After your verdict, explain your reason in one sentence.

Your response format must be exactly:
VERDICT: PASS or FAIL
REASON: your one sentence explanation
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Parse verdict and reason
    verdict = "FAIL"
    reason = "Could not parse critic response"

    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            verdict = "PASS" if "PASS" in line else "FAIL"
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    print(f"Critic verdict: {verdict}")
    print(f"Critic reason: {reason}")

    return {"verdict": verdict, "reason": reason}