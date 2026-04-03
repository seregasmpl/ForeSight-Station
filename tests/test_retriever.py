import numpy as np
import sys
import pytest
from indexer import IndexedCollection
from retriever import bm25_search, dense_search, rrf_merge, mmr_select, apply_dedup_penalty, HybridRetriever


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="sentence_transformers not compatible with Python 3.13")
def test_bm25_search(optimist_dir):
    col = IndexedCollection.build("optimist", optimist_dir, chunk_size=500, chunk_overlap=100)
    results = bm25_search("космос звёзды", col.fts_db, k=5)
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    chunk_ids = [r[0] for r in results]
    assert all(0 <= cid < len(col.chunks) for cid in chunk_ids)


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="sentence_transformers not compatible with Python 3.13")
def test_dense_search(optimist_dir):
    col = IndexedCollection.build("optimist", optimist_dir, chunk_size=500, chunk_overlap=100)
    results = dense_search("космические путешествия", col.embeddings, col.embed_model, k=5)
    assert len(results) > 0
    chunk_ids = [r[0] for r in results]
    scores = [r[1] for r in results]
    assert all(0 <= cid < len(col.chunks) for cid in chunk_ids)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_rrf_merge():
    bm25 = [(0, 1.0), (1, 0.8), (2, 0.6)]
    dense = [(1, 0.9), (3, 0.7), (0, 0.5)]
    merged = rrf_merge(bm25, dense, k=60)
    ids = [r[0] for r in merged]
    assert 1 in ids
    assert 0 in ids
    assert len(ids) == len(set(ids))


def test_mmr_select():
    np.random.seed(42)
    candidates = [(i, 1.0 - i * 0.1) for i in range(5)]
    embeddings = np.random.rand(5, 4).astype(np.float32)
    query_emb = np.random.rand(4).astype(np.float32)
    selected = mmr_select(candidates, embeddings, query_emb, k=3, lambda_param=0.7)
    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_apply_dedup_penalty():
    scores = [(0, 1.0), (1, 0.9), (2, 0.8)]
    used = {0, 2}
    adjusted = apply_dedup_penalty(scores, used, penalty=0.4)
    adj_dict = dict(adjusted)
    assert adj_dict[0] < 1.0
    assert adj_dict[1] == 0.9
    assert adj_dict[2] < 0.8


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="sentence_transformers not compatible with Python 3.13")
def test_hybrid_retriever_e2e(optimist_dir):
    col = IndexedCollection.build("optimist", optimist_dir, chunk_size=500, chunk_overlap=100)
    retriever = HybridRetriever(col)
    chunks = retriever.search("орбитальные станции и энергия звёзд", k=3, used_chunk_ids=set())
    assert 1 <= len(chunks) <= 3
    assert all(hasattr(c, "text") for c in chunks)
