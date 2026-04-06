from __future__ import annotations

import os
import sqlite3
import numpy as np
import Stemmer

from models import Chunk

_stemmer = Stemmer.Stemmer("russian")


def stem_russian(text: str) -> str:
    words = text.lower().split()
    stemmed = _stemmer.stemWords(words)
    return " ".join(stemmed)


def load_texts(directory: str) -> list[dict]:
    results = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            results.append({"text": text, "source_file": filename})
    return results


def chunk_texts(
    texts: list[dict], chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_id = 0
    for item in texts:
        raw = item["text"]
        source = item["source_file"]
        if len(raw) <= chunk_size:
            chunks.append(Chunk(id=chunk_id, text=raw, source_file=source, collection=""))
            chunk_id += 1
            continue
        start = 0
        while start < len(raw):
            end = start + chunk_size
            chunk_text = raw[start:end]
            chunks.append(Chunk(id=chunk_id, text=chunk_text, source_file=source, collection=""))
            chunk_id += 1
            start += chunk_size - overlap
    return chunks


def build_fts_index(chunks: list[Chunk]) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(stemmed_text, tokenize='unicode61')"
    )
    for chunk in chunks:
        stemmed = stem_russian(chunk.text)
        db.execute("INSERT INTO chunks_fts(rowid, stemmed_text) VALUES (?, ?)", (chunk.id, stemmed))
    db.commit()
    return db


def build_dense_index(
    chunks: list[Chunk], model_name: str = "intfloat/multilingual-e5-small"
) -> tuple[np.ndarray, object]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [f"passage: {chunk.text}" for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32), model


def _cache_path(directory: str, suffix: str) -> str:
    return os.path.join(directory, f"_cache_{suffix}")


class IndexedCollection:
    def __init__(
        self,
        name: str,
        chunks: list[Chunk],
        fts_db: sqlite3.Connection,
        embeddings: np.ndarray,
        embed_model: object,
    ):
        self.name = name
        self.chunks = chunks
        self.fts_db = fts_db
        self.embeddings = embeddings
        self.embed_model = embed_model

    def save_cache(self, directory: str):
        """Save pre-built embeddings and chunks to disk for fast startup."""
        import json
        np.save(_cache_path(directory, "embeddings.npy"), self.embeddings)
        chunks_data = [{"id": c.id, "text": c.text, "source_file": c.source_file, "collection": c.collection} for c in self.chunks]
        with open(_cache_path(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

    @classmethod
    def load_cache(cls, name: str, directory: str, model_name: str = "intfloat/multilingual-e5-small") -> IndexedCollection | None:
        """Load pre-built index from cache. Returns None if cache is missing or stale."""
        import json
        emb_path = _cache_path(directory, "embeddings.npy")
        chunks_path = _cache_path(directory, "chunks.json")
        if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
            return None

        # Check if any .txt file is newer than cache
        cache_mtime = os.path.getmtime(emb_path)
        for f in os.listdir(directory):
            if f.endswith(".txt") and os.path.getmtime(os.path.join(directory, f)) > cache_mtime:
                return None  # stale cache

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = [Chunk(**d) for d in chunks_data]
        embeddings = np.load(emb_path)
        fts_db = build_fts_index(chunks)

        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(model_name)

        return cls(name, chunks, fts_db, embeddings, embed_model)

    @classmethod
    def build(
        cls,
        name: str,
        directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> IndexedCollection:
        # Try loading from cache first
        cached = cls.load_cache(name, directory, model_name)
        if cached is not None:
            return cached

        texts = load_texts(directory)
        chunks = chunk_texts(texts, chunk_size, chunk_overlap)
        for chunk in chunks:
            chunk.collection = name
        fts_db = build_fts_index(chunks)
        embeddings, embed_model = build_dense_index(chunks, model_name)
        col = cls(name, chunks, fts_db, embeddings, embed_model)
        col.save_cache(directory)
        return col
