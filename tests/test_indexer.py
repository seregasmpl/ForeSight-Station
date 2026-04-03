import sqlite3
import numpy as np
import sys
import pytest
from indexer import load_texts, chunk_texts, stem_russian, build_fts_index, build_dense_index, IndexedCollection


def test_load_texts(optimist_dir):
    texts = load_texts(optimist_dir)
    assert len(texts) == 2
    assert all("source_file" in t and "text" in t for t in texts)
    assert any("Ефремов" in t["text"] for t in texts)


def test_chunk_texts():
    texts = [{"text": "А" * 1500, "source_file": "test.txt"}]
    chunks = chunk_texts(texts, chunk_size=1000, overlap=200)
    assert len(chunks) >= 2
    # Overlap: second chunk starts at 800, so first 200 chars of chunk 2 == last 200 of chunk 1
    assert chunks[0].text[-200:] == chunks[1].text[:200]


def test_chunk_texts_short():
    texts = [{"text": "Короткий текст", "source_file": "short.txt"}]
    chunks = chunk_texts(texts, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert chunks[0].text == "Короткий текст"


def test_stem_russian():
    stemmed = stem_russian("космонавты летали на орбитальных станциях")
    assert "космонавт" in stemmed or "косм" in stemmed
    words = stemmed.split()
    assert len(words) == 5


def test_build_fts_index(optimist_dir):
    texts = load_texts(optimist_dir)
    chunks = chunk_texts(texts, chunk_size=1000, overlap=200)
    db = build_fts_index(chunks)
    cursor = db.execute(
        "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 5",
        (stem_russian("космос звёзды"),),
    )
    results = cursor.fetchall()
    assert len(results) > 0


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="sentence_transformers not compatible with Python 3.13")
def test_build_dense_index(optimist_dir):
    texts = load_texts(optimist_dir)
    chunks = chunk_texts(texts, chunk_size=1000, overlap=200)
    embeddings, model = build_dense_index(chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0


@pytest.mark.skipif(sys.version_info >= (3, 13), reason="sentence_transformers not compatible with Python 3.13")
def test_indexed_collection(optimist_dir):
    collection = IndexedCollection.build("optimist", optimist_dir)
    assert len(collection.chunks) > 0
    assert collection.fts_db is not None
    assert collection.embeddings.shape[0] == len(collection.chunks)
