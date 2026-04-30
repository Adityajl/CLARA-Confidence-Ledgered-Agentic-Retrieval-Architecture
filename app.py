"""
CLARA – app.py
Streamlit UI — mirrors the look/feel of agentic_rag's app_llama3.2.py
but adds a live confidence dashboard panel.

Run: streamlit run app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from clara_rag import CLARARetriever
from confidence_engine import ConfidenceLedger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLARA – Confidence-Ledgered Agentic RAG",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .conf-high   { color: #1a7f4e; font-weight: 600; }
    .conf-mid    { color: #b07d00; font-weight: 600; }
    .conf-low    { color: #c0392b; font-weight: 600; }
    .step-badge  { background: #e8f4f8; border-radius: 4px;
                   padding: 2px 8px; font-size: 12px; color: #1a5276; }
    .ledger-stat { font-size: 13px; color: #555; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar config (mirrors agentic_rag sidebar) ─────────────────────────────
st.sidebar.title("⚙️ CLARA Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence threshold", 0.4, 0.95, 0.72, 0.01,
    help="Stop retrieving when aggregate confidence reaches this level"
)
max_steps = st.sidebar.slider("Max retrieval steps", 1, 8, 4)
top_k = st.sidebar.slider("Chunks per step (top-k)", 2, 10, 5)
model = st.sidebar.selectbox("Ollama model", ["llama3.2", "deepseek-r1", "mistral"])
enable_web = st.sidebar.toggle("Web fallback (FireCrawl)", value=False)
qdrant_url = st.sidebar.text_input("Qdrant URL", ":memory:")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Ingest documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt, .md or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
)

# ── Session state ──────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False

def get_retriever():
    if st.session_state.retriever is None:
        st.session_state.retriever = CLARARetriever(
            collection_name="clara_docs",
            qdrant_url=qdrant_url,
            confidence_thresh=confidence_threshold,
            max_steps=max_steps,
            top_k=top_k,
            model=model,
            enable_web_fallback=enable_web,
        )
    return st.session_state.retriever

# ── Document ingestion ────────────────────────────────────────────────────────
if uploaded_files and not st.session_state.ingested:
    retriever = get_retriever()
    docs = []
    for f in uploaded_files:
        if f.name.endswith(".pdf"):
            import fitz  # pymupdf
            doc = fitz.open(stream=f.read(), filetype="pdf")
            text = "\n\n".join(page.get_text() for page in doc)
        else:
            if f.name.endswith(".pdf"):
                import fitz  # pymupdf
                doc = fitz.open(stream=f.read(), filetype="pdf")
                text = "\n\n".join(page.get_text() for page in doc)
            else:
                text = f.read().decode("utf-8", errors="ignore")
        # Simple paragraph chunking (mirrors chonkie semantic chunking)
        chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        for chunk in chunks:
            docs.append({"text": chunk, "metadata": {"source": f.name}})
    retriever.ingest_documents(docs)
    st.session_state.ingested = True
    st.sidebar.success(f"✅ Ingested {len(docs)} chunks from {len(uploaded_files)} files")

# ── Main layout ────────────────────────────────────────────────────────────────
col_chat, col_dash = st.columns([3, 2])

with col_chat:
    st.title("🔬 CLARA")
    st.caption("Confidence-Ledgered Agentic Retrieval Architecture")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "trace" in msg:
                trace = msg["trace"]
                conf = trace.overall_confidence
                color_cls = "conf-high" if conf >= 0.72 else "conf-mid" if conf >= 0.50 else "conf-low"
                st.markdown(
                    f'<span class="{color_cls}">Confidence: {conf:.2f}</span> &nbsp;'
                    f'<span class="step-badge">Steps: {trace.retrieval_steps}</span> &nbsp;'
                    f'{"⚡ Early exit" if trace.terminated_early else ""}',
                    unsafe_allow_html=True,
                )

    # Query input
    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        retriever = get_retriever()
        with st.chat_message("assistant"):
            with st.spinner("Retrieving with confidence gating..."):
                trace = retriever.run(query, verbose=False)

            st.markdown(trace.final_answer or "_No answer generated — check Ollama is running._")

            conf = trace.overall_confidence
            color_cls = "conf-high" if conf >= 0.72 else "conf-mid" if conf >= 0.50 else "conf-low"
            st.markdown(
                f'<span class="{color_cls}">Confidence: {conf:.2f}</span> &nbsp;'
                f'<span class="step-badge">Steps: {trace.retrieval_steps}/{max_steps}</span> &nbsp;'
                f'{"⚡ Early exit — saved steps!" if trace.terminated_early else ""}',
                unsafe_allow_html=True,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": trace.final_answer,
            "trace": trace,
        })
        st.rerun()

with col_dash:
    st.subheader("📊 Confidence Dashboard")

    # Live ledger stats
    ledger = ConfidenceLedger()
    stats = ledger.get_stats()

    m1, m2 = st.columns(2)
    m1.metric("Total sessions", stats["total_sessions"])
    m2.metric("Avg confidence", f"{stats['avg_confidence']:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("Claims stored", stats["total_claims"])
    m4.metric("Early exit rate", f"{stats['early_exit_rate']*100:.0f}%")

    st.markdown("---")

    # Latest trace breakdown
    if st.session_state.messages:
        last_assistant = next(
            (m for m in reversed(st.session_state.messages)
             if m["role"] == "assistant" and "trace" in m),
            None,
        )
        if last_assistant:
            trace = last_assistant["trace"]
            st.markdown("**Last retrieval — claim confidence scores**")
            for i, claim in enumerate(trace.claims[:8]):
                conf = claim.confidence
                bar_color = "#1a7f4e" if conf >= 0.72 else "#b07d00" if conf >= 0.5 else "#c0392b"
                bar_pct = int(conf * 100)
                st.markdown(
                    f'<div style="margin-bottom:6px">'
                    f'<div style="font-size:11px;color:#555;margin-bottom:2px">'
                    f'[Step {claim.retrieval_step}] {claim.text[:70]}{"..." if len(claim.text)>70 else ""}'
                    f'</div>'
                    f'<div style="background:#eee;border-radius:4px;height:8px">'
                    f'<div style="width:{bar_pct}%;background:{bar_color};height:8px;border-radius:4px"></div>'
                    f'</div>'
                    f'<div style="font-size:10px;color:{bar_color};text-align:right">{conf:.3f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("**Retrieval trace**")
            st.json({
                "session_id": trace.session_id,
                "query": trace.query[:80],
                "steps_used": trace.retrieval_steps,
                "max_steps": max_steps,
                "overall_confidence": round(trace.overall_confidence, 3),
                "terminated_early": trace.terminated_early,
                "claims_count": len(trace.claims),
                "tokens_estimate": trace.tokens_used,
            })
    else:
        st.info("Ask a question to see live confidence scores here.")
