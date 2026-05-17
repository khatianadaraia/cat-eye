"""
Dual-stage retrieval: bi-encoder candidate selection + cross-encoder reranking.
"""

import numpy as np


class SkeletonSearch:
    def __init__(self):
        self._bi_encoder = None
        self._cross_encoder = None

    @property
    def bi_encoder(self):
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._bi_encoder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._cross_encoder

    def search(self, query, chunks, embeddings, top_k=5, n_candidates=30):
        from sentence_transformers import util

        query_embedding = self.bi_encoder.encode([query], convert_to_numpy=True)
        scores = util.cos_sim(query_embedding, embeddings).flatten().numpy()

        k_candidates = min(n_candidates, len(scores))
        top_indices = np.argsort(scores)[::-1][:k_candidates]

        pairs = [[query, chunks[idx]["text"]] for idx in top_indices]
        rerank_scores = self.cross_encoder.predict(pairs)
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

    def map_skeleton(self, subtopics, chunks, embeddings, top_k=5, n_candidates=30):
        seen_chunks = set()
        results_map = {}

        for subtopic in subtopics:
            results = self.search(subtopic, chunks, embeddings, top_k=top_k, n_candidates=n_candidates)
            for r in results:
                chunk_key = (r["source"], r["page"], r["text"][:80])
                r["duplicate"] = chunk_key in seen_chunks
                seen_chunks.add(chunk_key)
            results_map[subtopic] = results

        return results_map

    def coverage_report(self, results_map, folder_path):
        import os
        source_counts = {}
        node_scores = []

        for subtopic, results in results_map.items():
            scores = [r["score"] for r in results]
            node_scores.append({
                "subtopic": subtopic,
                "avg_score": np.mean(scores) if scores else 0,
                "top_score": max(scores) if scores else 0,
                "n_results": len(results)
            })
            for r in results:
                source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

        all_pdfs = set(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))
        cited = set(source_counts.keys())
        uncited = all_pdfs - cited

        return {
            "source_frequency": sorted(source_counts.items(), key=lambda x: -x[1]),
            "node_scores": node_scores,
            "uncited_papers": sorted(uncited)
        }
