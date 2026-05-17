import os
import sys
import re
import json
import hashlib
import argparse
import numpy as np
from pypdf import PdfReader

DEFAULT_SUBTOPICS = [
    "HPA axis calibration cortisol damage to hippocampus episodic working memory",
    "Allostatic load childhood poverty stress mediating pathway cognitive capacity",
    "Pace of Ageing biological ageing rate multi-system physiological decline biomarkers",
    "Parental education childhood SES cognitive functioning old age life course epidemiology"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Academic Skeleton Miner: Deterministic semantic search against your PDF library."
    )
    parser.add_argument("--folder", "-f", type=str, default=None,
                        help="Path to folder containing PDF files.")
    parser.add_argument("--output", "-o", type=str, default="literature_review_index.html",
                        help="Output HTML file path.")
    parser.add_argument("--subtopics", "-s", type=str, default=None,
                        help="Path to JSON/YAML file containing subtopics list.")
    parser.add_argument("--top-k", "-k", type=int, default=3,
                        help="Number of top matches to display per skeleton node.")
    parser.add_argument("--candidates", "-c", type=int, default=20,
                        help="Number of bi-encoder candidates to pass to cross-encoder.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable embedding cache (re-encode all documents).")
    return parser.parse_args()


def load_subtopics(path):
    if path is None:
        return DEFAULT_SUBTOPICS

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(f)
        elif ext == ".json":
            data = json.load(f)
        else:
            print(f"[ERROR] Unsupported subtopics file format: {ext} (use .json or .yaml)")
            sys.exit(1)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "subtopics" in data:
        return data["subtopics"]
    print("[ERROR] Subtopics file must be a list or a dict with a 'subtopics' key.")
    sys.exit(1)


def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)


def extract_and_chunk_pdfs(folder_path):
    all_chunks = []
    if not os.path.exists(folder_path):
        print(f"[ERROR] Specified folder target not found: {folder_path}")
        sys.exit(1)

    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print(f"[ERROR] No PDF files located in directory: {folder_path}")
        sys.exit(1)

    print(f" -> Mapping files: found {len(pdf_files)} target documents.")
    for filename in pdf_files:
        pdf_path = os.path.join(folder_path, filename)
        try:
            reader = PdfReader(pdf_path)
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
                            all_chunks.append({
                                "source": filename,
                                "page": page_num,
                                "text": chunk_text
                            })
                        current_chunk = []
                        current_word_count = 0
                    current_chunk.append(sentence)
                    current_word_count += words_in_sentence

                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text.split()) > 30:
                        all_chunks.append({
                            "source": filename,
                            "page": page_num,
                            "text": chunk_text
                        })
        except Exception as e:
            print(f"      [!] Skipping corrupted file {filename}: {e}")
    return all_chunks


def compute_folder_hash(folder_path):
    hasher = hashlib.md5()
    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        hasher.update(filename.encode())
        hasher.update(str(os.path.getmtime(filepath)).encode())
        hasher.update(str(os.path.getsize(filepath)).encode())
    return hasher.hexdigest()


def get_cache_path(folder_path):
    folder_hash = compute_folder_hash(folder_path)
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"embeddings_{folder_hash}.npz")


def load_cached_embeddings(cache_path):
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["chunks"].tolist()
    return None, None


def save_cached_embeddings(cache_path, embeddings, chunks):
    np.savez(cache_path, embeddings=embeddings, chunks=np.array(chunks, dtype=object))


def extract_keywords(subtopic):
    stop_words = {"the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with", "is", "are", "was", "were"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', subtopic.lower())
    return [w for w in words if w not in stop_words]


def highlight_keywords(text, keywords):
    highlighted = text
    for kw in keywords:
        pattern = re.compile(r'(\b' + re.escape(kw) + r'\b)', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark>\1</mark>', highlighted)
    return highlighted


def main():
    args = parse_args()
    print("=== ACADEMIC SKELETON MINER: STARTING EXTRACTION ===")

    folder_path = args.folder
    if folder_path is None:
        folder_path = r"C:\Users\khatn\OneDrive\Escritorio\Introduction-06-05-2026"

    subtopics = load_subtopics(args.subtopics)
    print(f" -> Loaded {len(subtopics)} skeleton nodes.")

    my_chunks = extract_and_chunk_pdfs(folder_path)
    total_chunks = len(my_chunks)
    print(f" -> Text mapping complete: Isolated {total_chunks} search targets.")

    print("\n -> Building semantic coordinate vector index via Sentence-Transformers...")
    from sentence_transformers import SentenceTransformer, util
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_texts = [chunk["text"] for chunk in my_chunks]

    doc_embeddings = None
    if not args.no_cache:
        cache_path = get_cache_path(folder_path)
        cached_embeddings, cached_chunks = load_cached_embeddings(cache_path)
        if cached_embeddings is not None and len(cached_chunks) == len(my_chunks):
            chunks_match = all(
                c["source"] == my_chunks[i]["source"] and c["page"] == my_chunks[i]["page"]
                for i, c in enumerate(cached_chunks)
            )
            if chunks_match:
                print("    [CACHE HIT] Loading pre-computed embeddings.")
                doc_embeddings = cached_embeddings

    if doc_embeddings is None:
        doc_embeddings = embedding_model.encode(
            corpus_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True
        )
        if not args.no_cache:
            save_cached_embeddings(cache_path, doc_embeddings, my_chunks)
            print("    [CACHE SAVED] Embeddings stored for future runs.")

    print("\n -> Initializing neural cross-attention model for re-ranking...")
    from sentence_transformers import CrossEncoder
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("\n -> Pre-encoding all skeleton nodes...")
    query_embeddings = embedding_model.encode(subtopics, convert_to_numpy=True)

    print("\n -> Processing nodes step-by-step...")
    section_evidence_map = {}
    seen_chunks = set()

    for num, subtopic in enumerate(subtopics):
        print(f"    Running match indexing for Node {num+1}/{len(subtopics)}...")

        query_embedding = query_embeddings[num]
        scores = util.cos_sim(query_embedding, doc_embeddings).flatten().numpy()

        k_candidates = min(args.candidates, len(scores))
        top_indices = np.argsort(scores)[::-1][:k_candidates]

        pairs = [[subtopic, my_chunks[idx]["text"]] for idx in top_indices]
        rerank_scores = rerank_model.predict(pairs)
        ranked_order = np.argsort(rerank_scores)[::-1]

        section_evidence_map[subtopic] = []
        collected = 0
        for rank_pos in ranked_order:
            if collected >= args.top_k:
                break
            idx = top_indices[rank_pos]
            chunk_key = (my_chunks[idx]["source"], my_chunks[idx]["page"], my_chunks[idx]["text"][:80])

            is_duplicate = chunk_key in seen_chunks
            seen_chunks.add(chunk_key)

            section_evidence_map[subtopic].append({
                "source_file": my_chunks[idx]["source"],
                "page": my_chunks[idx]["page"],
                "raw_text": my_chunks[idx]["text"],
                "score": float(rerank_scores[rank_pos]),
                "duplicate": is_duplicate
            })
            collected += 1

    build_html(args.output, section_evidence_map, subtopics)

    print("\n" + "=" * 60)
    print(" SKELETON INDEX COMPILED WITH ZERO HALLUCINATIONS! ")
    print(f" Saved dashboard to path: '{args.output}' ")
    print("=" * 60)


def build_html(output_file, section_evidence_map, subtopics):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Academic Skeleton Evidence Index</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292f; max-width: 940px; margin: 40px auto; padding: 0 20px; background-color: #ffffff; }
        h1 { font-size: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; color: #0969da; }
        .desc-tag { color: #57606a; font-size: 13.5px; margin-bottom: 20px; line-height: 1.5; }
        .search-box { width: 100%; padding: 10px 14px; font-size: 14px; border: 1px solid #d0d7de; border-radius: 6px; margin-bottom: 24px; box-sizing: border-box; }
        .search-box:focus { outline: none; border-color: #0969da; box-shadow: 0 0 0 3px rgba(9,105,218,0.1); }
        .node-section { margin-bottom: 28px; }
        .node-header { font-size: 17px; font-weight: 600; color: #1f2328; border-bottom: 1px solid #d8dee4; padding: 8px 0; cursor: pointer; user-select: none; display: flex; align-items: center; gap: 8px; }
        .node-header:hover { color: #0969da; }
        .collapse-icon { transition: transform 0.2s; font-size: 12px; }
        .collapsed .collapse-icon { transform: rotate(-90deg); }
        .collapsed .node-body { display: none; }
        .node-body { padding-top: 12px; }
        .evidence-block { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 16px; margin-bottom: 14px; position: relative; }
        .evidence-block.duplicate { border-left: 3px solid #bf8700; }
        .meta-line { font-size: 12px; font-weight: 600; color: #24292f; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .source-file-title { color: #cf222e; }
        .page-ref { color: #0969da; font-weight: 500; }
        .score-badge { background: #ddf4ff; color: #0969da; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .score-badge.low { background: #fff8c5; color: #9a6700; }
        .score-badge.negative { background: #ffebe9; color: #cf222e; }
        .duplicate-badge { background: #fff8c5; color: #9a6700; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-left: 6px; }
        .quote-body { font-style: normal; font-size: 14px; text-align: justify; color: #24292f; }
        mark { background: #fff8c5; padding: 1px 2px; border-radius: 2px; }
        .stats { font-size: 12px; color: #57606a; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Academic Skeleton Cross-Reference Index</h1>
    <div class="desc-tag">
        <strong>Forensic Extraction Verified:</strong> This tool completely bypasses the generative AI slope. It indexes physical document text directly against skeleton nodes using dual-stage neural keyword matching to protect academic integrity.
    </div>
    <input type="text" class="search-box" id="searchBox" placeholder="Filter evidence blocks by keyword..." oninput="filterResults()">
    <div class="stats" id="stats"></div>
    <div id="content">
""")

        for num, subtopic in enumerate(subtopics, 1):
            records = section_evidence_map[subtopic]
            keywords = extract_keywords(subtopic)

            f.write(f'    <div class="node-section" data-node="{num}">\n')
            f.write(f'        <div class="node-header" onclick="toggleSection(this)">')
            f.write(f'<span class="collapse-icon">&#9660;</span> Skeleton Node {num}: {subtopic}</div>\n')
            f.write(f'        <div class="node-body">\n')

            for rank, rec in enumerate(records, 1):
                score = rec["score"]
                score_class = "negative" if score < 0 else ("low" if score < 3 else "")
                dup_class = " duplicate" if rec["duplicate"] else ""
                dup_badge = '<span class="duplicate-badge">ALSO MATCHED ABOVE</span>' if rec["duplicate"] else ""
                highlighted_text = highlight_keywords(rec["raw_text"], keywords)

                f.write(f'        <div class="evidence-block{dup_class}">\n')
                f.write(f'            <div class="meta-line">')
                f.write(f'<span>[Rank {rank}] <span class="source-file-title">{rec["source_file"]}</span>')
                f.write(f' &mdash; <span class="page-ref">p.{rec["page"]}</span></span>')
                f.write(f' <span><span class="score-badge {score_class}">{score:.2f}</span>{dup_badge}</span>')
                f.write(f'</div>\n')
                f.write(f'            <div class="quote-body">&ldquo;{highlighted_text}&rdquo;</div>\n')
                f.write(f'        </div>\n')

            f.write(f'        </div>\n')
            f.write(f'    </div>\n')

        f.write("""    </div>
    <script>
        function toggleSection(header) {
            header.parentElement.classList.toggle('collapsed');
        }
        function filterResults() {
            const query = document.getElementById('searchBox').value.toLowerCase();
            const blocks = document.querySelectorAll('.evidence-block');
            let visible = 0;
            blocks.forEach(block => {
                const text = block.textContent.toLowerCase();
                const match = !query || text.includes(query);
                block.style.display = match ? '' : 'none';
                if (match) visible++;
            });
            document.getElementById('stats').textContent = query ? visible + ' results matching "' + query + '"' : '';
        }
    </script>
</body>
</html>""")


if __name__ == "__main__":
    main()
