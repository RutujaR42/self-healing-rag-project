import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.vectorstore import load_vectorstore, retrieve_chunks
from src.generator import generate_answer
from src.critic import critic_agent

load_dotenv()

# ── 1. Define the pipeline state ──────────────────────────────────────────────
# This is the "memory" of the pipeline — everything passed between steps
class RAGState(TypedDict):
    question:        str         # original user question
    refined_question: str        # reformulated question on retry
    chunks:          list        # retrieved document chunks
    answer:          str         # generated answer
    verdict:         str         # PASS or FAIL from critic
    reason:          str         # critic's explanation
    retries:         int         # how many times we've retried


# ── 2. Define each node (step) in the graph ───────────────────────────────────

def retrieve_node(state: RAGState) -> RAGState:
    """Step 1 — Retrieve relevant chunks from ChromaDB."""
    question = state.get("refined_question") or state["question"]
    print(f"\n[RETRIEVE] Searching for: '{question}'")
    vectorstore = load_vectorstore()
    chunks = retrieve_chunks(question, vectorstore)
    return {**state, "chunks": chunks}


def generate_node(state: RAGState) -> RAGState:
    """Step 2 — Generate answer from retrieved chunks."""
    question = state.get("refined_question") or state["question"]
    print(f"[GENERATE] Generating answer...")
    answer = generate_answer(question, state["chunks"])
    return {**state, "answer": answer}


def critic_node(state: RAGState) -> RAGState:
    """Step 3 — Critique the answer for hallucinations."""
    print(f"[CRITIC] Evaluating answer...")
    result = critic_agent(
        state["question"],
        state["answer"],
        state["chunks"]
    )
    return {
        **state,
        "verdict": result["verdict"],
        "reason":  result["reason"]
    }


def refine_query_node(state: RAGState) -> RAGState:
    """Step 4 — Reformulate the query if critic said FAIL."""
    retries = state.get("retries", 0) + 1
    original = state["question"]
    refined  = f"Explain in detail with examples: {original}"
    print(f"[REFINE] Retry #{retries} — New query: '{refined}'")
    return {**state, "refined_question": refined, "retries": retries}


def fallback_node(state: RAGState) -> RAGState:
    """Step 5 — Give up gracefully after max retries."""
    print("[FALLBACK] Max retries reached. Returning fallback response.")
    return {
        **state,
        "answer": "I don't have enough information in my knowledge base to answer this question accurately."
    }


# ── 3. Decision logic — what happens after the critic ─────────────────────────

def should_retry(state: RAGState) -> str:
    """Decide next step based on critic verdict and retry count."""
    if state["verdict"] == "PASS":
        print("[DECISION] Answer approved! ✅")
        return "approved"
    if state.get("retries", 0) >= 2:
        print("[DECISION] Max retries hit. Going to fallback.")
        return "fallback"
    print("[DECISION] Answer rejected. Retrying...")
    return "retry"


# ── 4. Build the LangGraph pipeline ───────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic",   critic_node)
    graph.add_node("refine",   refine_query_node)
    graph.add_node("fallback", fallback_node)

    # Define the flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")

    # After critic — branch based on verdict
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "approved": END,
            "retry":    "refine",
            "fallback": "fallback"
        }
    )

    # After refine — go back to retrieve (the retry loop!)
    graph.add_edge("refine", "retrieve")
    graph.add_edge("fallback", END)

    return graph.compile()


# ── 5. Main run function ───────────────────────────────────────────────────────

def run_pipeline(question: str) -> str:
    """Run the full self-healing RAG pipeline."""
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    pipeline = build_graph()

    initial_state = RAGState(
        question=question,
        refined_question="",
        chunks=[],
        answer="",
        verdict="",
        reason="",
        retries=0
    )

    final_state = pipeline.invoke(initial_state)

    print(f"\n{'='*50}")
    print(f"FINAL ANSWER: {final_state['answer']}")
    print(f"VERDICT: {final_state['verdict']}")
    print(f"RETRIES: {final_state['retries']}")
    print(f"{'='*50}")

    return final_state["answer"]