"""
Skeleton Miner — Deterministic semantic evidence extraction for academic research.
Zero hallucination. Every word comes from your documents.
Copyright (c) 2026 Khatia Nadaraia. All rights reserved.
"""

import os
import re
import json
import tempfile
import hashlib
import numpy as np
import streamlit as st
from pathlib import Path
from pypdf import PdfReader

st.set_page_config(page_title="Skeleton Miner", page_icon="🦴", layout="wide", initial_sidebar_state="expanded")

CACHE_DIR = Path(tempfile.gettempdir()) / "skeletonminer_cache"
CACHE_DIR.mkdir(exist_ok=True)


# --- Core Engine ---

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)


def extract_chunks_from_pdfs(pdf_files_data):
    all_chunks = []
    for filename, file_bytes in pdf_files_data:
        try:
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if not text or len(text.strip()) < 50:
                    continue
                sentences = split_sentences(text.strip())
                current_chunk = []
                current_word_count = 0
                for sentence in sentences:
                    words_in_sentence = len(sentence.split())
                    if current_word_count + words_in_sentence > 200 and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        if len(chunk_text.split()) > 30:
                            all_chunks.append({"source": filename, "page": page_num, "text": chunk_text})
                        current_chunk = []
                        current_word_count = 0
                    current_chunk.append(sentence)
                    current_word_count += words_in_sentence
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text.split()) > 30:
                        all_chunks.append({"source": filename, "page": page_num, "text": chunk_text})
        except Exception:
            continue
    return all_chunks


def extract_chunks_from_folder(folder_path):
    pdf_files_data = []
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, f)
            with open(filepath, "rb") as fh:
                pdf_files_data.append((f, fh.read()))
    return extract_chunks_from_pdfs(pdf_files_data)


def compute_hash(pdf_files_data):
    hasher = hashlib.md5()
    for name, data in pdf_files_data:
        hasher.update(name.encode())
        hasher.update(str(len(data)).encode())
    return hasher.hexdigest()


@st.cache_resource
def load_bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_data
def encode_chunks(chunks_texts, _hash_key):
    bi_encoder = load_bi_encoder()
    embeddings = bi_encoder.encode(chunks_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def search(query, chunks, embeddings, top_k=5, n_candidates=30):
    from sentence_transformers import util
    bi_encoder = load_bi_encoder()
    cross_encoder = load_cross_encoder()

    query_embedding = bi_encoder.encode([query], convert_to_numpy=True)
    scores = util.cos_sim(query_embedding, embeddings).flatten().numpy()

    k_candidates = min(n_candidates, len(scores))
    top_indices = np.argsort(scores)[::-1][:k_candidates]

    pairs = [[query, chunks[idx]["text"]] for idx in top_indices]
    rerank_scores = cross_encoder.predict(pairs)
    ranked_order = np.argsort(rerank_scores)[::-1]

    results = []
    for pos in ranked_order[:top_k]:
        idx = top_indices[pos]
        results.append({
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "score": float(rerank_scores[pos])
        })
    return results


def map_skeleton(subtopics, chunks, embeddings, top_k=5, n_candidates=30):
    seen_chunks = set()
    results_map = {}
    for subtopic in subtopics:
        results = search(subtopic, chunks, embeddings, top_k=top_k, n_candidates=n_candidates)
        for r in results:
            chunk_key = (r["source"], r["page"], r["text"][:80])
            r["duplicate"] = chunk_key in seen_chunks
            seen_chunks.add(chunk_key)
        results_map[subtopic] = results
    return results_map


def extract_keywords(text):
    stop_words = {"the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with", "is", "are", "was", "were", "that", "this", "been", "have", "has", "had", "does", "did", "but", "not", "from", "they", "its", "can", "may", "more", "than", "between"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in stop_words]


def highlight_text(text, keywords):
    for kw in keywords:
        pattern = re.compile(r'(\b' + re.escape(kw) + r'\b)', re.IGNORECASE)
        text = pattern.sub(r'**\1**', text)
    return text


# --- UI Styling ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f8f9fb; }
    .main-title { font-size: 2.4rem; font-weight: 800; color: #0f172a; margin-bottom: 0; letter-spacing: -0.5px; }
    .subtitle { font-size: 1rem; color: #64748b; margin-top: 4px; margin-bottom: 28px; line-height: 1.5; }
    .result-card { background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #3b82f6; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
    .result-card-weak { background: #fffbeb; border: 1px solid #fde68a; border-left: 3px solid #f59e0b; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
    .result-card-bad { background: #fef2f2; border: 1px solid #fecaca; border-left: 3px solid #ef4444; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
    .meta-line { font-size: 0.78rem; color: #475569; margin-bottom: 6px; font-weight: 500; }
    .source-tag { background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .page-tag { background: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .score-tag { padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
    .score-high { background: #dcfce7; color: #166534; }
    .score-mid { background: #fef9c3; color: #854d0e; }
    .score-low { background: #fee2e2; color: #991b1b; }
    .dup-tag { background: #fef9c3; color: #92400e; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; }
    .node-title { font-size: 1.05rem; font-weight: 700; color: #1e293b; padding: 10px 0 6px 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 12px; }
    .gap-alert { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 8px 12px; font-size: 0.82rem; color: #92400e; margin-bottom: 12px; }
    .hero-box { background: linear-gradient(135deg, #f0f9ff 0%, #f0fdf4 100%); border: 1px solid #bae6fd; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .how-it-works { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🦴 Skeleton Miner")
    st.caption("v0.1.0 — by Khatia Nadaraia")
    st.markdown("---")

    mode = st.radio("Mode", ["📤 Upload PDFs", "📁 Local folder"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Settings**")
    top_k = st.slider("Results per query", 1, 10, 5)
    n_candidates = st.slider("Candidate pool", 10, 50, 30)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.6;">
    <strong>How it works:</strong><br>
    1. Bi-encoder finds candidate passages<br>
    2. Cross-encoder reranks by deep relevance<br>
    3. Output: exact original text only<br><br>
    <strong>Zero generation. Zero hallucination.</strong><br>
    Every word comes from your documents.
    </div>
    """, unsafe_allow_html=True)


# --- Main ---
st.markdown('<div class="main-title">Skeleton Miner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">The anti-ChatGPT for academic writing. Upload your papers, define your argument skeleton, and extract exact evidence — no AI writing, no hallucination, just forensic retrieval.</div>', unsafe_allow_html=True)


# --- Load PDFs ---
chunks = None
embeddings = None

if mode == "📤 Upload PDFs":
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload your research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="All processing happens in-session. Your files are never stored permanently."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        pdf_files_data = [(f.name, f.read()) for f in uploaded_files]
        with st.spinner(f"Indexing {len(pdf_files_data)} documents..."):
            chunks = extract_chunks_from_pdfs(pdf_files_data)
            if chunks:
                corpus_texts = [c["text"] for c in chunks]
                hash_key = compute_hash(pdf_files_data)
                embeddings = encode_chunks(corpus_texts, hash_key)
        if chunks:
            st.success(f"✓ {len(pdf_files_data)} PDFs indexed — {len(chunks)} text segments ready for search.")

else:
    folder_path = st.text_input("PDF folder path", placeholder=r"C:\path\to\your\pdfs")
    if folder_path and os.path.isdir(folder_path):
        pdf_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
        if pdf_count > 0:
            with st.spinner(f"Indexing {pdf_count} documents from folder..."):
                chunks = extract_chunks_from_folder(folder_path)
                if chunks:
                    corpus_texts = [c["text"] for c in chunks]
                    hash_key = hashlib.md5(folder_path.encode()).hexdigest()
                    embeddings = encode_chunks(corpus_texts, hash_key)
            if chunks:
                st.success(f"✓ {pdf_count} PDFs indexed — {len(chunks)} text segments ready.")
        else:
            st.warning("No PDF files found in that folder.")
    elif folder_path:
        st.error("Folder not found. Check the path.")


# --- If no data loaded, show explanation ---
if chunks is None or embeddings is None:
    st.markdown("---")
    st.markdown('<div class="how-it-works">', unsafe_allow_html=True)
    st.markdown("""
    **What this tool does:**

    You have a folder of research papers and a paper skeleton (your section headings).
    This tool maps your entire library against your skeleton structure — showing you
    exactly which passages from which papers support each section.

    **What makes it different from ChatGPT / NotebookLM / Anara:**

    - ❌ It does NOT generate text
    - ❌ It does NOT summarise
    - ❌ It does NOT paraphrase
    - ✅ It shows you the **exact original words** from the **exact page** of the **exact paper**
    - ✅ It scores confidence so you know if a match is strong or weak
    - ✅ It detects gaps where your library doesn't cover a topic

    **Upload your PDFs above to try it.**
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# --- Tabs ---
tab_search, tab_skeleton, tab_coverage = st.tabs(["🔍 Live Search", "🦴 Skeleton Map", "📊 Coverage"])

# === LIVE SEARCH ===
with tab_search:
    st.markdown("Type any research question. Results come exclusively from your uploaded documents.")
    query = st.text_input("Search your library", placeholder="Does childhood poverty accelerate biological ageing through HPA axis dysregulation?", label_visibility="collapsed")

    if query:
        with st.spinner("Searching..."):
            results = search(query, chunks, embeddings, top_k=top_k, n_candidates=n_candidates)
        keywords = extract_keywords(query)

        for i, r in enumerate(results, 1):
            score = r["score"]
            card_class = "result-card" if score > 3 else ("result-card-weak" if score > 0 else "result-card-bad")
            score_class = "score-high" if score > 3 else ("score-mid" if score > 0 else "score-low")

            st.markdown(f"""<div class="{card_class}">
                <div class="meta-line">
                    #{i} &nbsp;&nbsp;
                    <span class="source-tag">{r['source']}</span> &nbsp;
                    <span class="page-tag">p.{r['page']}</span> &nbsp;
                    <span class="score-tag {score_class}">{score:.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            highlighted = highlight_text(r["text"], keywords)
            st.markdown(f"> {highlighted}")
            st.markdown("")


# === SKELETON MAP ===
with tab_skeleton:
    st.markdown("Define your paper's argument structure — one topic per line. The engine maps your entire library against it and shows where your evidence is strong or weak.")

    subtopics_text = st.text_area(
        "Skeleton nodes (one per line)",
        height=200,
        placeholder="free word recall episodic memory early marker Alzheimer's disease\nHPA axis calibration cortisol chronic stress hippocampus\nallostatic load childhood poverty mediating pathway working memory\n..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🔬 Analyze", type="primary")

    if run_btn:
        subtopics = [s.strip() for s in subtopics_text.strip().split("\n") if s.strip()]
        if not subtopics:
            st.warning("Enter at least one skeleton node.")
        else:
            with st.spinner(f"Mapping {len(subtopics)} skeleton nodes..."):
                results_map = map_skeleton(subtopics, chunks, embeddings, top_k=top_k, n_candidates=n_candidates)
                st.session_state["skeleton_results"] = results_map

            st.success(f"✓ {len(subtopics)} nodes mapped — {len(subtopics) * top_k} evidence blocks extracted.")

    if "skeleton_results" in st.session_state:
        results_map = st.session_state["skeleton_results"]

        for num, (subtopic, results) in enumerate(results_map.items(), 1):
            keywords = extract_keywords(subtopic)
            avg_score = np.mean([r["score"] for r in results]) if results else 0
            strength = "🟢" if avg_score > 3 else ("🟡" if avg_score > 1 else "🔴")

            st.markdown(f'<div class="node-title">{strength} Node {num}: {subtopic}</div>', unsafe_allow_html=True)

            if avg_score < 1:
                st.markdown(f'<div class="gap-alert">⚠️ Weak coverage — your library may not have enough papers on this topic.</div>', unsafe_allow_html=True)

            for i, r in enumerate(results, 1):
                score = r["score"]
                card_class = "result-card" if score > 3 else ("result-card-weak" if score > 0 else "result-card-bad")
                score_class = "score-high" if score > 3 else ("score-mid" if score > 0 else "score-low")
                dup_html = ' &nbsp;<span class="dup-tag">DUPLICATE</span>' if r.get("duplicate") else ""

                st.markdown(f"""<div class="{card_class}">
                    <div class="meta-line">
                        #{i} &nbsp;&nbsp;
                        <span class="source-tag">{r['source']}</span> &nbsp;
                        <span class="page-tag">p.{r['page']}</span> &nbsp;
                        <span class="score-tag {score_class}">{score:.2f}</span>{dup_html}
                    </div>
                </div>""", unsafe_allow_html=True)

                highlighted = highlight_text(r["text"], keywords)
                st.markdown(f"> {highlighted}")


# === COVERAGE ===
with tab_coverage:
    st.markdown("### How well does your library cover your argument?")

    if "skeleton_results" in st.session_state:
        results_map = st.session_state["skeleton_results"]

        source_counts = {}
        for subtopic, results in results_map.items():
            for r in results:
                source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Most-cited papers")
            sorted_sources = sorted(source_counts.items(), key=lambda x: -x[1])
            for source, count in sorted_sources[:12]:
                bar = "█" * min(count, 30)
                st.code(f"{count:2d} │ {bar} │ {source}", language=None)

        with col_right:
            st.markdown("#### Evidence strength per node")
            for i, (subtopic, results) in enumerate(results_map.items(), 1):
                scores = [r["score"] for r in results]
                avg = np.mean(scores) if scores else 0
                top = max(scores) if scores else 0
                icon = "🟢" if avg > 3 else ("🟡" if avg > 1 else "🔴")
                st.code(f"{icon} Node {i:2d} │ avg={avg:.1f} top={top:.1f} │ {subtopic[:45]}", language=None)

        all_cited = set(source_counts.keys())
        all_sources = set(r["source"] for results in results_map.values() for r in results)
        uncited_sources = set()
        if mode == "📁 Local folder" and folder_path:
            all_pdfs = set(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))
            uncited_sources = all_pdfs - all_cited

        if uncited_sources:
            st.markdown("#### Papers not appearing in any result")
            st.caption("These are in your library but didn't match any node.")
            for f in sorted(uncited_sources):
                st.text(f"   ○ {f}")
    else:
        st.info("Run a Skeleton Map analysis first to see coverage data.")
