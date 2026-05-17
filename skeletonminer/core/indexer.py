"""
PDF document indexer with sentence-aware chunking and embedding cache.
"""

import os
import re
import hashlib
import numpy as np
from pathlib import Path
from pypdf import PdfReader


class PDFIndexer:
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._bi_encoder = None

    @property
    def bi_encoder(self):
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._bi_encoder

    def index_folder(self, folder_path, force_reindex=False):
        folder_path = str(folder_path)
        chunks = self._extract_chunks(folder_path)
        if not chunks:
            return [], None

        if not force_reindex:
            cached = self._load_cache(folder_path, chunks)
            if cached is not None:
                return chunks, cached

        corpus_texts = [c["text"] for c in chunks]
        embeddings = self.bi_encoder.encode(
            corpus_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True
        )
        self._save_cache(folder_path, embeddings, chunks)
        return chunks, embeddings

    def _extract_chunks(self, folder_path):
        all_chunks = []
        if not os.path.isdir(folder_path):
            return all_chunks

        pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
        for filename in pdf_files:
            pdf_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if not text or len(text.strip()) < 50:
                        continue
                    page_chunks = self._chunk_text(text.strip(), filename, page_num)
                    all_chunks.extend(page_chunks)
            except Exception:
                continue
        return all_chunks

    def _chunk_text(self, text, filename, page_num):
        chunks = []
        sentences = self._split_sentences(text)
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            words_in_sentence = len(sentence.split())
            if current_word_count + words_in_sentence > 200 and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) > 30:
                    chunks.append({"source": filename, "page": page_num, "text": chunk_text})
                current_chunk = []
                current_word_count = 0
            current_chunk.append(sentence)
            current_word_count += words_in_sentence

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) > 30:
                chunks.append({"source": filename, "page": page_num, "text": chunk_text})
        return chunks

    def _split_sentences(self, text):
        return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    def _folder_hash(self, folder_path):
        hasher = hashlib.md5()
        pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
        for filename in pdf_files:
            filepath = os.path.join(folder_path, filename)
            hasher.update(filename.encode())
            hasher.update(str(os.path.getmtime(filepath)).encode())
            hasher.update(str(os.path.getsize(filepath)).encode())
        return hasher.hexdigest()

    def _cache_path(self, folder_path):
        return self.cache_dir / f"embeddings_{self._folder_hash(folder_path)}.npz"

    def _load_cache(self, folder_path, chunks):
        cache_path = self._cache_path(folder_path)
        if not cache_path.exists():
            return None
        data = np.load(str(cache_path), allow_pickle=True)
        cached_embeddings = data["embeddings"]
        cached_chunks = data["chunks"].tolist()
        if len(cached_chunks) != len(chunks):
            return None
        match = all(
            c["source"] == chunks[i]["source"] and c["page"] == chunks[i]["page"]
            for i, c in enumerate(cached_chunks)
        )
        return cached_embeddings if match else None

    def _save_cache(self, folder_path, embeddings, chunks):
        cache_path = self._cache_path(folder_path)
        np.savez(str(cache_path), embeddings=embeddings, chunks=np.array(chunks, dtype=object))
