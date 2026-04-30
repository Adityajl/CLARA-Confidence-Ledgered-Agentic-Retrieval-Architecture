# CLARA — Confidence-Ledgered Agentic Retrieval Architecture

> **Built on top of [`patchy631/ai-engineering-hub/agentic_rag`](https://github.com/patchy631/ai-engineering-hub/tree/main/agentic_rag)**
> Same stack (CrewAI · Qdrant · fastembed · Ollama · Streamlit) — with confidence gating added.

---

## What CLARA adds to `agentic_rag`

| Capability | `agentic_rag` | **CLARA** |
|---|---|---|
| Document retrieval | ✅ Fixed k steps | ✅ Confidence-gated early exit |
| Confidence scoring | ❌ None | ✅ Per-claim scorer (3 signals) |
| Memory across sessions | ❌ None | ✅ SQLite Confidence Ledger |
| Web fallback | ✅ FireCrawl | ✅ FireCrawl (same) |
| Observability | ❌ None | ✅ Live dashboard |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Ollama (same as `agentic_rag`)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2        # or deepseek-r1
```

### 3. Run the demo (no Ollama needed)

```bash
cd clara
python demo.py
```

### 4. Run the full Streamlit app

```bash
streamlit run app.py
```

Upload `.txt` or `.md` files in the sidebar, then ask questions.
The confidence dashboard shows per-claim scores and early-exit savings in real time.

---

## Environment variables (`.env`)

```env
# Required only for cloud Qdrant (optional — in-memory works by default)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_api_key

# Required only for web fallback (same as agentic_rag)
FIRECRAWL_API_KEY=your_firecrawl_key
```

---

## Project structure

```
clara/
├── app.py                      # Streamlit UI (replaces app_llama3.2.py)
├── demo.py                     # Quick demo — no Ollama needed
├── requirements.txt
└── src/
    ├── confidence_engine.py    # ConfidenceScorer · ConfidenceLedger · Claim
    └── clara_rag.py            # CLARARetriever (wraps agentic_rag loop)
```

---

## How the confidence gate works

```
Query
  │
  ▼
[Ledger warm-start] ── cached high-confidence claims? ──► add to pool
  │
  ▼
For step in 1..max_steps:
  │
  ├─ Retrieve top-k chunks from Qdrant
  ├─ Generate partial answer (Ollama)
  ├─ Extract claims from answer
  ├─ Score each claim:
  │     signal_1 = cosine(claim_embed, chunk_embeds)   # semantic fit
  │     signal_2 = unique_sources / total_chunks       # source diversity
  │     signal_3 = 1 - avg_rank * 0.15                # retrieval rank
  │     confidence = 0.55·s1 + 0.25·s2 + 0.20·s3
  │
  └─ Aggregate = harmonic mean of all claims
        if aggregate >= threshold  →  STOP EARLY ⚡
        else                       →  next step
  │
  ▼
[Web fallback if aggregate < 0.50 and web enabled]
  │
  ▼
Final answer  +  save trace to Confidence Ledger
```

---

## Paper roadmap

This is Month 1 of the CLARA research plan:

- **Month 1 (this code):** Confidence head + benchmark design
- **Month 2:** RL-trained graph memory policy (Neo4j + TRL PPO)
- **Month 3:** Full eval + paper write-up (NeurIPS 2026 target)

---

*From [`patchy631/ai-engineering-hub`](https://github.com/patchy631/ai-engineering-hub) with confidence.*
