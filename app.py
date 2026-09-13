"""
IEEE RAS AI Knowledge Assistant - Streamlit Application
A modern, retrieval-augmented intelligence interface for IEEE Robotics and Automation Society.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

from src.config import DEFAULT_GEMINI_MODEL, EMBEDDING_MODEL_NAME
from src.retriever import get_retriever
from src.rag import get_rag_pipeline

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IEEE RAS AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Modern Dark Robotics / AI Theme CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Custom Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(30, 41, 59, 0.4) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-badge {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: rgba(56, 189, 248, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 20px;
        padding: 4px 10px;
        vertical-align: middle;
        margin-left: 10px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .header-disclaimer {
        color: #64748b;
        font-size: 0.82rem;
        font-style: italic;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .sidebar-tagline {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 18px;
        line-height: 1.4;
    }
    .sidebar-section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #38bdf8;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 4px;
    }

    /* Pipeline Cards in Sidebar */
    .pipeline-step {
        display: flex;
        align-items: center;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .step-number {
        background: #0284c7;
        color: #ffffff;
        font-weight: 700;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        margin-right: 10px;
    }
    .step-text {
        font-weight: 500;
        color: #cbd5e1;
    }

    /* Knowledge Base Metric Card */
    .kb-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 14px;
        font-size: 0.85rem;
    }

    /* Welcome Hero Cards */
    .welcome-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
    }
    .welcome-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .welcome-icon {
        font-size: 1.6rem;
        margin-bottom: 8px;
    }
    .welcome-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .welcome-card-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.45;
    }

    /* Source Citation Badges */
    .source-badge {
        display: inline-block;
        background: rgba(2, 132, 199, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 6px;
        padding: 6px 12px;
        margin: 4px;
        font-size: 0.85rem;
        text-decoration: none;
    }
    .source-badge:hover {
        background: rgba(2, 132, 199, 0.25);
        color: #7dd3fc;
    }

    /* Chat bubble polish */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# App State & Pipeline Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Retrieve retriever & pipeline instances
retriever = get_retriever()
env_key = os.getenv("GEMINI_API_KEY", "")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 IEEE RAS AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-tagline">Explore IEEE Robotics and Automation Society through a retrieval-augmented AI assistant.</div>',
        unsafe_allow_html=True
    )

    # Quick Questions Section
    st.markdown('<div class="sidebar-section-title">Quick Questions</div>', unsafe_allow_html=True)
    
    quick_questions = [
        "What is IEEE RAS?",
        "What are the main technical committees?",
        "What conferences are associated with IEEE RAS?",
        "How can students get involved?",
        "What publications does IEEE RAS support?",
        "What is the mission of IEEE RAS?"
    ]

    for q in quick_questions:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    # RAG Pipeline Steps
    st.markdown('<div class="sidebar-section-title">RAG Pipeline</div>', unsafe_allow_html=True)
    pipeline_steps = [
        ("1", "Retrieve", "FAISS Vector Search"),
        ("2", "Rank", "Cosine Relevance"),
        ("3", "Generate", "Grounded Gemini LLM"),
        ("4", "Cite", "Verified Source URLs")
    ]
    for num, name, detail in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <span class="step-number">{num}</span>
            <div>
                <div class="step-text">{name}</div>
                <div style="font-size: 0.72rem; color: #64748b;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Knowledge Base Telemetry
    st.markdown('<div class="sidebar-section-title">Knowledge Base</div>', unsafe_allow_html=True)
    total_vectors = retriever.index.ntotal if retriever.is_ready() else 0
    st.markdown(f"""
    <div class="kb-card">
        <div style="color: #38bdf8; font-weight: 600; margin-bottom: 4px;">Public IEEE RAS Web Information</div>
        <div style="color: #94a3b8;">• Chunks Indexed: <b style="color: #f1f5f9;">{total_vectors}</b></div>
        <div style="color: #94a3b8;">• Embeddings: <b style="color: #f1f5f9;">{EMBEDDING_MODEL_NAME}</b></div>
        <div style="color: #94a3b8;">• Status: <span style="color: #10b981; font-weight: 600;">● Active</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Clear Chat Button
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# Instantiate RAG Pipeline with environment key
pipeline = get_rag_pipeline(api_key=env_key)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="header-title">
        IEEE RAS AI Knowledge Assistant
        <span class="header-badge">RAG Powered</span>
    </div>
    <div class="header-subtitle">
        Ask questions about IEEE Robotics and Automation Society using publicly available information.
    </div>
    <div class="header-disclaimer">
        Community-built RAG assistant using publicly available IEEE RAS information. Not an official IEEE RAS service.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Empty State / Feature Highlights
# -----------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("### 👋 Welcome to the IEEE RAS Knowledge Assistant")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🎯</div>
            <div class="welcome-card-title">Strictly Grounded</div>
            <div class="welcome-card-desc">
                Answers exclusively using retrieved official IEEE RAS documentation to avoid hallucinations and fabricated details.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">⚡</div>
            <div class="welcome-card-title">FAISS Vector Search</div>
            <div class="welcome-card-desc">
                High-performance semantic retrieval indexed over IEEE RAS committees, publications, conferences, and student initiatives.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🔗</div>
            <div class="welcome-card-title">Source Attribution</div>
            <div class="welcome-card-desc">
                Every generated response includes transparent relevance scores, retrieved document excerpts, and clickable source links.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Render Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Display RAG Debug / Retrieved Sources expander for assistant messages
        if role == "assistant" and msg.get("chunks"):
            with st.expander("🔍 Retrieved Sources & Transparency", expanded=False):
                st.caption(f"Top Similarity Score: **{msg.get('similarity_score', 0.0) * 100:.1f}%** | Chunks Retrieved: **{len(msg.get('chunks', []))}**")
                for i, chunk in enumerate(msg.get("chunks", []), 1):
                    score_pct = chunk.get("similarity_score", 0.0) * 100
                    st.markdown(f"**[{i}] [{chunk.get('title', 'Source')}]({chunk.get('url', '#')})** — *Relevance: {score_pct:.1f}%*")
                    st.info(f"\"{chunk.get('text', '')[:260]}...\"")

        # Display clean clickable sources cards
        if role == "assistant" and msg.get("sources"):
            st.markdown("**Sources:**")
            sources_html = ""
            for s in msg.get("sources", []):
                sources_html += f'<a class="source-badge" href="{s["url"]}" target="_blank">🌐 {s["title"]}</a> '
            st.markdown(sources_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Chat Input & Processing
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about IEEE RAS conferences, committees, publications...")

# Handle sidebar button prompt click
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate response via RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching IEEE RAS knowledge base and generating answer..."):
            result = pipeline.answer_question(user_query)
            answer_text = result["answer"]
            chunks = result.get("retrieved_chunks", [])
            sources = result.get("sources", [])
            similarity_score = result.get("similarity_score", 0.0)

            st.markdown(answer_text)

            # RAG transparency expander
            if chunks:
                with st.expander("🔍 Retrieved Sources & Transparency", expanded=False):
                    st.caption(f"Top Similarity Score: **{similarity_score * 100:.1f}%** | Chunks Retrieved: **{len(chunks)}**")
                    for i, chunk in enumerate(chunks, 1):
                        score_pct = chunk.get("similarity_score", 0.0) * 100
                        st.markdown(f"**[{i}] [{chunk.get('title', 'Source')}]({chunk.get('url', '#')})** — *Relevance: {score_pct:.1f}%*")
                        st.info(f"\"{chunk.get('text', '')[:260]}...\"")

            # Clickable sources
            if sources:
                st.markdown("**Sources:**")
                sources_html = ""
                for s in sources:
                    sources_html += f'<a class="source-badge" href="{s["url"]}" target="_blank">🌐 {s["title"]}</a> '
                st.markdown(sources_html, unsafe_allow_html=True)

    # 3. Save assistant message to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "chunks": chunks,
        "sources": sources,
        "similarity_score": similarity_score
    })
