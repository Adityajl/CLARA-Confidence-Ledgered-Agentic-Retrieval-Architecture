"""
CLARA – Confidence-Ledgered Agentic Retrieval Architecture
confidence_engine.py

Wraps the agentic_rag retrieval loop with:
  1. Per-claim confidence scoring (entropy-based + semantic consistency)
  2. Confidence-gated early exit (stop retrieving when confident enough)
  3. Confidence Ledger — persists scores to SQLite across sessions
"""

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """A single factual claim extracted from a retrieval step."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    source_chunk: str = ""
    source_doc: str = ""
    confidence: float = 0.0          # 0.0 – 1.0
    retrieval_step: int = 0
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.05         # confidence decays per day if not reinforced

    def decayed_confidence(self) -> float:
        days_old = (time.time() - self.timestamp) / 86400
        return max(0.0, self.confidence * math.exp(-self.decay_rate * days_old))


@dataclass
class RetrievalTrace:
    """Full trace of one agentic retrieval session."""
    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    claims: list[Claim] = field(default_factory=list)
    retrieval_steps: int = 0
    tokens_used: int = 0
    final_answer: str = ""
    overall_confidence: float = 0.0
    terminated_early: bool = False


# ── Confidence Scorer ─────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Lightweight confidence estimator that works WITHOUT a fine-tuned model.
    Uses three signals that are available from any LLM API response:

    1. Semantic consistency  — check if the chunk text actually entails the claim
                               via embedding cosine similarity (fastembed)
    2. Source diversity      — penalize if all top-k chunks come from one doc
    3. Retrieval rank signal — higher-ranked chunks → higher base confidence

    These three signals are combined into a single [0, 1] score.
    In the paper version this would be replaced by a LoRA confidence head.
    """

    def __init__(self, similarity_threshold: float = 0.72):
        self.similarity_threshold = similarity_threshold
        self._embed_model = None  # lazy-load

    def _get_embedder(self):
        if self._embed_model is None:
            try:
                from fastembed import TextEmbedding
                self._embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:
                # Fallback: keyword-overlap scoring (no model download needed)
                self._embed_model = "unavailable"
        return self._embed_model

    def _cosine(self, a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

    def score_claim(
        self,
        claim_text: str,
        supporting_chunks: list[dict],   # [{"text": str, "doc": str, "rank": int}]
        retrieval_step: int = 0,
    ) -> Claim:
        """Score one claim against its supporting chunks."""
        if not supporting_chunks:
            return Claim(text=claim_text, confidence=0.1, retrieval_step=retrieval_step)

        embedder = self._get_embedder()

        # ── Signal 1: semantic consistency ────────────────────────────────────
        if embedder != "unavailable":
            try:
                vecs = list(embedder.embed([claim_text] + [c["text"] for c in supporting_chunks]))
                claim_vec = vecs[0]
                chunk_vecs = vecs[1:]
                sims = [self._cosine(claim_vec, cv) for cv in chunk_vecs]
                sem_score = float(np.mean(sims))
            except Exception:
                sem_score = 0.5
        else:
            # Fallback: keyword overlap ratio
            claim_words = set(claim_text.lower().split())
            overlaps = []
            for c in supporting_chunks:
                chunk_words = set(c["text"].lower().split())
                overlap = len(claim_words & chunk_words) / max(len(claim_words), 1)
                overlaps.append(overlap)
            sem_score = float(np.mean(overlaps)) if overlaps else 0.3

        # ── Signal 2: source diversity (penalize single-source over-reliance) ─
        docs = [c.get("doc", "unknown") for c in supporting_chunks]
        unique_docs = len(set(docs))
        diversity_score = min(1.0, unique_docs / max(len(supporting_chunks), 1))

        # ── Signal 3: retrieval rank (rank 0 = best match) ───────────────────
        avg_rank = float(np.mean([c.get("rank", 3) for c in supporting_chunks]))
        rank_score = max(0.0, 1.0 - avg_rank * 0.15)  # penalty per rank position

        # ── Combine (weighted average) ────────────────────────────────────────
        confidence = (
            0.55 * sem_score +
            0.25 * diversity_score +
            0.20 * rank_score
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        best_chunk = supporting_chunks[0]
        return Claim(
            text=claim_text,
            source_chunk=best_chunk["text"][:200],
            source_doc=best_chunk.get("doc", "unknown"),
            confidence=confidence,
            retrieval_step=retrieval_step,
        )

    def should_stop(
        self,
        claims: list[Claim],
        confidence_threshold: float = 0.72,
        min_claims: int = 2,
    ) -> tuple[bool, float]:
        """
        Decide if the retrieval loop should terminate.
        Returns (should_stop, aggregate_confidence).
        """
        if len(claims) < min_claims:
            return False, 0.0

        # Aggregate = harmonic mean (pessimistic — one weak claim tanks the score)
        scores = [c.confidence for c in claims]
        harmonic = len(scores) / sum(1.0 / max(s, 1e-9) for s in scores)
        aggregate = float(np.clip(harmonic, 0.0, 1.0))

        return aggregate >= confidence_threshold, aggregate


# ── Confidence Ledger (SQLite persistence) ────────────────────────────────────

class ConfidenceLedger:
    """
    Persists claims and retrieval traces across sessions.
    Think of this as the 'memory' of what the system is confident about.
    """

    def __init__(self, db_path: str = "clara_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    source_chunk TEXT,
                    source_doc TEXT,
                    confidence REAL,
                    retrieval_step INTEGER,
                    timestamp REAL,
                    decay_rate REAL,
                    session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    query TEXT,
                    retrieval_steps INTEGER,
                    tokens_used INTEGER,
                    final_answer TEXT,
                    overall_confidence REAL,
                    terminated_early INTEGER,
                    created_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_claims_session ON claims(session_id);
                CREATE INDEX IF NOT EXISTS idx_claims_confidence ON claims(confidence);
            """)

    def save_trace(self, trace: RetrievalTrace):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?)
            """, (
                trace.session_id, trace.query, trace.retrieval_steps,
                trace.tokens_used, trace.final_answer, trace.overall_confidence,
                int(trace.terminated_early), time.time()
            ))
            for claim in trace.claims:
                conn.execute("""
                    INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    claim.id, claim.text, claim.source_chunk, claim.source_doc,
                    claim.confidence, claim.retrieval_step,
                    claim.timestamp, claim.decay_rate, trace.session_id
                ))

    def get_high_confidence_claims(
        self, query_keywords: list[str], threshold: float = 0.70, limit: int = 5
    ) -> list[Claim]:
        """Retrieve previously-confident claims relevant to current query."""
        with sqlite3.connect(self.db_path) as conn:
            # Simple keyword match — in production use vector search
            placeholders = " OR ".join(["text LIKE ?" for _ in query_keywords])
            params = [f"%{kw}%" for kw in query_keywords] + [threshold]
            rows = conn.execute(f"""
                SELECT id, text, source_chunk, source_doc, confidence,
                       retrieval_step, timestamp, decay_rate
                FROM claims
                WHERE ({placeholders}) AND confidence >= ?
                ORDER BY confidence DESC LIMIT {limit}
            """, params).fetchall()

        claims = []
        for row in rows:
            c = Claim(
                id=row[0], text=row[1], source_chunk=row[2], source_doc=row[3],
                confidence=row[4], retrieval_step=row[5],
                timestamp=row[6], decay_rate=row[7]
            )
            # Apply temporal decay before returning
            c.confidence = c.decayed_confidence()
            if c.confidence >= threshold:
                claims.append(c)
        return claims

    def get_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM claims").fetchone()[0] or 0
            early_exits = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE terminated_early=1"
            ).fetchone()[0]
        return {
            "total_sessions": total_sessions,
            "total_claims": total_claims,
            "avg_confidence": round(avg_conf, 3),
            "early_exits": early_exits,
            "early_exit_rate": round(early_exits / max(total_sessions, 1), 2),
        }
