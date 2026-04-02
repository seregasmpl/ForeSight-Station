from __future__ import annotations

import sqlite3
import numpy as np

from indexer import stem_russian, IndexedCollection
from models import Chunk


def bm25_search(query: str, fts_db: sqlite3.Connection, k: int = 50) -> list[tuple[int, float]]:
    stemmed_query = stem_russian(query)
    try:
        cursor = fts_db.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (stemmed_query, k),
        )
        results = cursor.fetchall()
    except sqlite3.OperationalError:
        return []
    if not results:
        return []
    max_neg = max(abs(r[1]) for r in results)
    if max_neg == 0:
        return [(r[0], 1.0) for r in results]
    return [(r[0], 1.0 - abs(r[1]) / (max_neg + 1e-9)) for r in results]


def dense_search(
    query: str,
    embeddings: np.ndarray,
    embed_model: object,
    k: int = 50,
) -> list[tuple[int, float]]:
    query_emb = embed_model.encode(
        [f"query: {query}"], normalize_embeddings=True
    )[0].astype(np.float32)
    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def rrf_merge(
    bm25_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = {}
    for rank, (chunk_id, _) in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def apply_dedup_penalty(
    scores: list[tuple[int, float]],
    used_chunk_ids: set[int],
    penalty: float = 0.4,
) -> list[tuple[int, float]]:
    adjusted = []
    for chunk_id, score in scores:
        if chunk_id in used_chunk_ids:
            adjusted.append((chunk_id, score - penalty))
        else:
            adjusted.append((chunk_id, score))
    return sorted(adjusted, key=lambda x: x[1], reverse=True)


def mmr_select(
    candidates: list[tuple[int, float]],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 10,
    lambda_param: float = 0.7,
) -> list[int]:
    if len(candidates) == 0:
        return []
    selected: list[int] = []
    remaining = {cid: score for cid, score in candidates}

    for _ in range(min(k, len(candidates))):
        best_id = -1
        best_mmr = -float("inf")
        for cid in remaining:
            relevance = remaining[cid]
            if selected:
                selected_embs = embeddings[selected]
                cid_emb = embeddings[cid]
                sims = selected_embs @ cid_emb
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_id = cid
        if best_id == -1:
            break
        selected.append(best_id)
        del remaining[best_id]
    return selected


class HybridRetriever:
    def __init__(self, collection: IndexedCollection):
        self.collection = collection

    def search(
        self, query: str, k: int = 10, used_chunk_ids: set[int] | None = None
    ) -> list[Chunk]:
        if used_chunk_ids is None:
            used_chunk_ids = set()

        bm25_results = bm25_search(query, self.collection.fts_db, k=50)
        dense_results = dense_search(
            query, self.collection.embeddings, self.collection.embed_model, k=50
        )
        merged = rrf_merge(bm25_results, dense_results)
        merged = apply_dedup_penalty(merged, used_chunk_ids)

        query_emb = self.collection.embed_model.encode(
            [f"query: {query}"], normalize_embeddings=True
        )[0].astype(np.float32)

        selected_ids = mmr_select(merged, self.collection.embeddings, query_emb, k=k)
        return [self.collection.chunks[cid] for cid in selected_ids]
