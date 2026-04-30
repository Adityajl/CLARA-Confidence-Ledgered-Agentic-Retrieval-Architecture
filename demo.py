"""
CLARA – demo.py
Quick demo that runs CLARA entirely locally with no external APIs needed.
Uses a small synthetic knowledge base instead of Ollama.

Run: python demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from confidence_engine import ConfidenceScorer, ConfidenceLedger, Claim
from clara_rag import CLARARetriever

# ── Synthetic knowledge base (no Ollama needed for this demo) ─────────────────
SAMPLE_DOCS = [
    {
        "text": "The Eiffel Tower is located in Paris, France. It was built by Gustave Eiffel "
                "and completed in 1889 for the World's Fair. It stands 330 meters tall.",
        "metadata": {"source": "landmarks.txt"},
    },
    {
        "text": "Paris is the capital city of France. It is known for its art, culture, "
                "and architecture. The population of Paris is approximately 2.1 million.",
        "metadata": {"source": "cities.txt"},
    },
    {
        "text": "The Louvre Museum is located in Paris and is the world's largest art museum. "
                "It houses the Mona Lisa and over 35,000 works of art.",
        "metadata": {"source": "museums.txt"},
    },
    {
        "text": "Machine learning is a subset of artificial intelligence. It enables systems "
                "to learn from data without being explicitly programmed.",
        "metadata": {"source": "ai_basics.txt"},
    },
    {
        "text": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
                "language model. The retrieval step fetches relevant documents, which are "
                "then used as context for the language model to generate an answer.",
        "metadata": {"source": "rag_explained.txt"},
    },
    {
        "text": "Confidence calibration in machine learning refers to how well the predicted "
                "probabilities of a model reflect the true likelihood of outcomes. "
                "A well-calibrated model's confidence scores match empirical accuracy.",
        "metadata": {"source": "calibration.txt"},
    },
]


def demo_confidence_scorer():
    print("\n" + "="*60)
    print("DEMO 1: Confidence Scorer")
    print("="*60)

    scorer = ConfidenceScorer()
    chunks = [
        {"text": SAMPLE_DOCS[0]["text"], "doc": "landmarks.txt", "rank": 0},
        {"text": SAMPLE_DOCS[1]["text"], "doc": "cities.txt", "rank": 1},
    ]

    claim_text = "The Eiffel Tower is in Paris and was built in 1889."
    claim = scorer.score_claim(claim_text, chunks, retrieval_step=0)

    print(f"Claim:      {claim.text}")
    print(f"Confidence: {claim.confidence:.3f}")
    print(f"Source:     {claim.source_doc}")

    # Test early-exit gate
    claims = [
        scorer.score_claim("Paris is the capital of France.", chunks),
        scorer.score_claim("The Eiffel Tower stands 330 meters tall.", chunks),
        scorer.score_claim("The Louvre is the world's largest art museum.", chunks),
    ]
    should_stop, agg = scorer.should_stop(claims, confidence_threshold=0.4)
    print(f"\nAggregate confidence (3 claims): {agg:.3f}")
    print(f"Should stop at threshold 0.40?   {'Yes ✓' if should_stop else 'No, keep retrieving'}")


def demo_ledger():
    print("\n" + "="*60)
    print("DEMO 2: Confidence Ledger (persistence)")
    print("="*60)

    ledger = ConfidenceLedger("demo_ledger.db")
    scorer = ConfidenceScorer()

    chunks = [{"text": SAMPLE_DOCS[0]["text"], "doc": "landmarks.txt", "rank": 0}]

    from confidence_engine import RetrievalTrace
    trace = RetrievalTrace(query="Where is the Eiffel Tower?")
    trace.final_answer = "The Eiffel Tower is in Paris, France."
    trace.overall_confidence = 0.81
    trace.terminated_early = True
    trace.retrieval_steps = 2
    trace.claims = [
        scorer.score_claim("The Eiffel Tower is in Paris.", chunks),
        scorer.score_claim("It was built in 1889.", chunks),
    ]

    ledger.save_trace(trace)
    print("Saved trace to ledger.")

    cached = ledger.get_high_confidence_claims(["eiffel", "paris"], threshold=0.3)
    print(f"Retrieved {len(cached)} cached claims for 'eiffel paris'")
    for c in cached:
        print(f"  [{c.confidence:.3f}] {c.text[:60]}")

    stats = ledger.get_stats()
    print(f"\nLedger stats: {stats}")


def demo_full_pipeline():
    print("\n" + "="*60)
    print("DEMO 3: Full CLARA retrieval pipeline")
    print("="*60)
    print("(Ollama not required — using chunk text as answer fallback)")

    retriever = CLARARetriever(
        collection_name="demo_collection",
        qdrant_url=":memory:",
        confidence_thresh=0.45,   # lower threshold for demo
        max_steps=3,
        top_k=3,
        model="llama3.2",         # won't be called if Ollama not running
        ledger_path="demo_ledger.db",
        enable_web_fallback=False,
    )

    # Ingest synthetic docs
    retriever.ingest_documents(SAMPLE_DOCS)

    # Run queries
    queries = [
        "Where is the Eiffel Tower located?",
        "What is Retrieval-Augmented Generation?",
        "Tell me about confidence calibration in ML.",
    ]

    results = []
    for q in queries:
        trace = retriever.run(q, verbose=True)
        results.append(trace)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Query':<42} {'Steps':>5} {'Conf':>6} {'Early?':>7}")
    print("-"*64)
    for t in results:
        print(
            f"{t.query[:41]:<42} {t.retrieval_steps:>5} "
            f"{t.overall_confidence:>6.3f} {'✓' if t.terminated_early else '✗':>7}"
        )

    # Ledger stats
    print()
    stats = retriever.ledger.get_stats()
    print(f"Ledger — sessions: {stats['total_sessions']}, "
          f"claims: {stats['total_claims']}, "
          f"avg conf: {stats['avg_confidence']}, "
          f"early exit rate: {stats['early_exit_rate']*100:.0f}%")


if __name__ == "__main__":
    demo_confidence_scorer()
    demo_ledger()
    demo_full_pipeline()
    print("\n✅ All demos complete. Run `streamlit run app.py` for the full UI.")
