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

    @classmethod
    def build(
        cls,
        name: str,
        directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> IndexedCollection:
        texts = load_texts(directory)
        chunks = chunk_texts(texts, chunk_size, chunk_overlap)
        for chunk in chunks:
            chunk.collection = name
        fts_db = build_fts_index(chunks)
        embeddings, embed_model = build_dense_index(chunks, model_name)
        return cls(name, chunks, fts_db, embeddings, embed_model)
