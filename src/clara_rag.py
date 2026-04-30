"""
CLARA – clara_rag.py

Drop-in replacement for the agentic_rag retrieval loop.
Adds confidence-gated early exit and ledger persistence on top of the
exact same CrewAI + Qdrant + Ollama stack from the repo.

Usage:
    from clara_rag import CLARARetriever
    retriever = CLARARetriever(collection_name="my_docs")
    result = retriever.run("What is the capital of France?")
    print(result.final_answer)
    print(f"Confidence: {result.overall_confidence:.2f}")
    print(f"Early exit: {result.terminated_early}")
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from confidence_engine import (
    ConfidenceScorer,
    ConfidenceLedger,
    Claim,
    RetrievalTrace,
)

load_dotenv()

# ── Optional Ollama LLM call (mirrors what app_llama3.2.py does) ─────────────
def _ollama_generate(prompt: str, model: str = "llama3.2", max_tokens: int = 512) -> str:
    """Call local Ollama. Returns empty string if Ollama is not running."""
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception:
        pass
    return ""


def _extract_claims_from_text(text: str) -> list[str]:
    """
    Naively split answer text into individual factual claims.
    In the paper version this is a structured extraction prompt.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ── CLARA Retriever ───────────────────────────────────────────────────────────

class CLARARetriever:
    """
    Confidence-Ledgered Agentic Retrieval.

    Wraps the agentic_rag retrieval loop with:
      - Confidence scoring at each retrieval step
      - Early exit when aggregate confidence >= threshold
      - Ledger persistence for cross-session memory
      - Web fallback (mirrors agentic_rag behaviour)

    Parameters
    ----------
    collection_name   : Qdrant collection (same as agentic_rag)
    qdrant_url        : Qdrant URL (default: in-memory for dev)
    confidence_thresh : Exit loop when aggregate confidence reaches this
    max_steps         : Hard cap on retrieval iterations
    top_k             : Chunks retrieved per step
    model             : Ollama model name
    ledger_path       : SQLite DB for confidence ledger
    enable_web_fallback: Fall back to web search (requires FireCrawl API key)
    """

    def __init__(
        self,
        collection_name: str = "clara_docs",
        qdrant_url: str = ":memory:",
        confidence_thresh: float = 0.72,
        max_steps: int = 4,
        top_k: int = 5,
        model: str = "llama3.2",
        ledger_path: str = "clara_ledger.db",
        enable_web_fallback: bool = False,
    ):
        self.collection_name = collection_name
        self.confidence_thresh = confidence_thresh
        self.max_steps = max_steps
        self.top_k = top_k
        self.model = model
        self.enable_web_fallback = enable_web_fallback

        # Core components
        self.scorer = ConfidenceScorer()
        self.ledger = ConfidenceLedger(ledger_path)

        # Qdrant client
        if qdrant_url == ":memory:":
            self.qdrant = QdrantClient(":memory:")
        else:
            self.qdrant = QdrantClient(
                url=qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY"),
            )

        # fastembed for retrieval (same as repo)
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
                self._embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:
                self._embedder = "unavailable"
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        if embedder == "unavailable":
            # TF-IDF style bag-of-words fallback (384-dim, sparse)
            import hashlib
            DIM = 384
            vecs = []
            for text in texts:
                vec = [0.0] * DIM
                for word in text.lower().split():
                    idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
                    vec[idx] += 1.0
                # L2 normalise
                norm = sum(v**2 for v in vec) ** 0.5 or 1.0
                vecs.append([v / norm for v in vec])
            return vecs
        return [v.tolist() for v in embedder.embed(texts)]

    def ingest_documents(self, docs: list[dict]):
        """
        Ingest documents into Qdrant.
        Each doc: {"text": str, "metadata": dict (optional)}
        Mirrors the chonkie[semantic] chunking step in agentic_rag.
        """
        # Ensure collection exists
        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        texts = [d["text"] for d in docs]
        vectors = self._embed(texts)

        points = [
            PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "text": docs[i]["text"],
                    "doc": docs[i].get("metadata", {}).get("source", f"doc_{i}"),
                },
            )
            for i in range(len(docs))
        ]
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f"[CLARA] Ingested {len(points)} chunks into '{self.collection_name}'")

    def _retrieve(self, query: str, step: int) -> list[dict]:
        """Retrieve top-k chunks from Qdrant for the current query."""
        query_vec = self._embed([query])[0]
        from qdrant_client.models import QueryRequest
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=self.top_k,
        ).points
        return [
            {
                "text": r.payload.get("text", ""),
                "doc": r.payload.get("doc", "unknown"),
                "rank": i,
                "score": r.score,
            }
            for i, r in enumerate(results)
        ]

    def _generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """Generate answer from retrieved context using Ollama."""
        context = "\n\n".join(
            f"[Source: {c['doc']}]\n{c['text']}" for c in context_chunks
        )
        prompt = f"""You are a helpful AI assistant. Use only the context below to answer.
If the context is insufficient, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        answer = _ollama_generate(prompt, model=self.model)
        if not answer:
            # Fallback: concatenate top chunk texts
            answer = " ".join(c["text"][:200] for c in context_chunks[:2])
        return answer.strip()

    def _web_fallback(self, query: str) -> str:
        """Mirrors agentic_rag FireCrawl web fallback."""
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return ""
        try:
            import requests
            resp = requests.post(
                "https://api.firecrawl.dev/v0/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"query": query, "limit": 3},
                timeout=15,
            )
            if resp.status_code == 200:
                results = resp.json().get("data", [])
                return "\n\n".join(r.get("markdown", "")[:500] for r in results)
        except Exception:
            pass
        return ""

    def run(self, query: str, verbose: bool = True) -> RetrievalTrace:
        """
        Run the confidence-gated agentic retrieval loop.

        Steps:
          1. Check ledger for existing high-confidence claims
          2. Retrieve → score → check threshold → maybe exit early
          3. Repeat up to max_steps
          4. Web fallback if confidence still low
          5. Persist to ledger
        """
        trace = RetrievalTrace(query=query)
        all_chunks: list[dict] = []
        all_claims: list[Claim] = []
        tokens_used = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"[CLARA] Query: {query}")
            print(f"[CLARA] Threshold: {self.confidence_thresh} | Max steps: {self.max_steps}")

        # ── Step 0: check ledger for warm-start ───────────────────────────────
        keywords = [w for w in query.lower().split() if len(w) > 3]
        cached_claims = self.ledger.get_high_confidence_claims(
            keywords, threshold=self.confidence_thresh + 0.05
        )
        if cached_claims:
            if verbose:
                print(f"[CLARA] Ledger warm-start: {len(cached_claims)} cached claims")
            all_claims.extend(cached_claims)

        # ── Retrieval loop ────────────────────────────────────────────────────
        for step in range(self.max_steps):
            trace.retrieval_steps = step + 1

            # Retrieve chunks
            chunks = self._retrieve(query, step)
            if not chunks:
                if verbose:
                    print(f"[CLARA] Step {step+1}: No chunks retrieved. Stopping.")
                break
            all_chunks.extend(chunks)

            # Generate partial answer from current chunks
            partial_answer = self._generate_answer(query, chunks)
            tokens_used += len(partial_answer.split()) * 2  # rough estimate

            # Extract and score claims from partial answer
            raw_claims = _extract_claims_from_text(partial_answer)
            step_claims = []
            for claim_text in raw_claims[:4]:  # cap per step to control cost
                claim = self.scorer.score_claim(claim_text, chunks, retrieval_step=step)
                step_claims.append(claim)
                all_claims.append(claim)

            # Check confidence gate
            should_stop, agg_confidence = self.scorer.should_stop(
                all_claims, self.confidence_thresh
            )

            if verbose:
                avg_step = sum(c.confidence for c in step_claims) / max(len(step_claims), 1)
                print(
                    f"[CLARA] Step {step+1}: {len(chunks)} chunks | "
                    f"step conf={avg_step:.3f} | agg conf={agg_confidence:.3f}"
                )

            if should_stop:
                trace.terminated_early = (step < self.max_steps - 1)
                if verbose:
                    print(f"[CLARA] ✓ Confidence threshold met at step {step+1}. Early exit!")
                break

        # ── Web fallback if still low confidence ─────────────────────────────
        _, final_confidence = self.scorer.should_stop(all_claims, 0.0)
        if final_confidence < 0.50 and self.enable_web_fallback:
            if verbose:
                print(f"[CLARA] Low confidence ({final_confidence:.2f}). Trying web fallback...")
            web_text = self._web_fallback(query)
            if web_text:
                web_chunk = {"text": web_text[:800], "doc": "web_search", "rank": 0}
                all_chunks.append(web_chunk)
                web_claims_text = _extract_claims_from_text(
                    self._generate_answer(query, [web_chunk])
                )
                for ct in web_claims_text[:3]:
                    claim = self.scorer.score_claim(ct, [web_chunk], retrieval_step=99)
                    all_claims.append(claim)

        # ── Final answer ──────────────────────────────────────────────────────
        final_answer = self._generate_answer(query, all_chunks[:self.top_k])
        _, overall_confidence = self.scorer.should_stop(all_claims, 0.0)

        trace.claims = all_claims
        trace.tokens_used = tokens_used
        trace.final_answer = final_answer
        trace.overall_confidence = overall_confidence

        # Persist to ledger
        self.ledger.save_trace(trace)

        if verbose:
            print(f"[CLARA] Final confidence: {overall_confidence:.3f}")
            print(f"[CLARA] Claims recorded: {len(all_claims)}")
            print(f"[CLARA] Tokens used: {tokens_used}")
            print(f"{'='*60}\n")

        return trace
