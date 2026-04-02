# Форсайт-Станция Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web application where 8 school teams interact with AI agents (techno-optimist / techno-pessimist) to prepare arguments for a foresight debate about space and humanity's future by 2100. Agents use RAG over sci-fi/popular-science text collections.

**Architecture:** Python FastAPI monolith serving static frontend. Hybrid search (BM25 via SQLite FTS5 + dense via multilingual-e5-small embeddings) over 2 text collections. OpenRouter API for LLM. In-memory session management for up to 8 concurrent teams. Soviet retro-futuristic UI.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, SQLite FTS5, sentence-transformers (multilingual-e5-small), PyStemmer, numpy, httpx, Pydantic, vanilla HTML/CSS/JS

---

## File Structure

```
SpaceGame/
├── serve.py                  # FastAPI app, all endpoints, startup indexing
├── config.py                 # All configuration constants
├── models.py                 # Pydantic schemas (Chunk, Session, Request, Response)
├── indexer.py                # Text loading, chunking, FTS5 + embedding indexing
├── retriever.py              # Hybrid search: BM25 + dense + RRF + MMR + dedup
├── agents.py                 # Prompt building, OpenRouter calls, response parsing
├── sessions.py               # Session creation, locking, heartbeat, history
├── requirements.txt
├── data/
│   ├── collections/
│   │   ├── optimist/         # .txt files for optimist agent
│   │   └── pessimist/        # .txt files for pessimist agent
├── static/
│   ├── index.html            # Entry page — session code input
│   ├── session.html          # Team workspace — chat with agent
│   ├── admin.html            # Teacher dashboard
│   ├── style.css             # Soviet retro-futurism styling
│   └── app.js                # All client-side logic
├── prompts/
│   ├── optimist.txt          # System prompt for techno-optimist
│   └── pessimist.txt         # System prompt for techno-pessimist
└── tests/
    ├── conftest.py           # Shared fixtures (sample texts, temp dirs)
    ├── test_indexer.py
    ├── test_retriever.py
    ├── test_agents.py
    ├── test_sessions.py
    ├── test_api.py
    └── fixtures/
        ├── optimist/
        │   ├── efremov_snippet.txt
        │   └── clarke_snippet.txt
        └── pessimist/
            ├── lem_snippet.txt
            └── bradbury_snippet.txt
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `data/collections/optimist/.gitkeep`
- Create: `data/collections/pessimist/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/optimist/efremov_snippet.txt`
- Create: `tests/fixtures/optimist/clarke_snippet.txt`
- Create: `tests/fixtures/pessimist/lem_snippet.txt`
- Create: `tests/fixtures/pessimist/bradbury_snippet.txt`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.6
uvicorn==0.34.0
pydantic==2.10.4
httpx==0.28.1
sentence-transformers==3.3.1
numpy==2.2.1
PyStemmer==2.2.0.3
pytest==8.3.4
pytest-asyncio==0.25.0
```

- [ ] **Step 2: Create config.py**

```python
import os

# --- Server ---
HOST = "0.0.0.0"
PORT = 8000

# --- LLM ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_TEMPERATURE = 0.8
LLM_TIMEOUT = 120.0

# --- Admin ---
ADMIN_PIN = os.environ.get("ADMIN_PIN", "1234")

# --- Indexing ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 200     # characters
COLLECTIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "collections")

# --- Retrieval ---
BM25_TOP_K = 50
DENSE_TOP_K = 50
RRF_K = 60
MMR_TOP_K = 10
MMR_LAMBDA = 0.7
DEDUP_PENALTY = 0.4

# --- Sessions ---
HEARTBEAT_INTERVAL = 30   # seconds (client sends)
HEARTBEAT_TIMEOUT = 120   # seconds (server considers disconnected)

# --- Topics ---
TOPICS = [
    "Человечество — космическая цивилизация",
    "Неисчерпаемые блага космоса",
    "Тайны космоса в наших руках",
    "Земля — наш космический корабль",
]

TOPIC_PREFIXES = {
    "Человечество — космическая цивилизация": "ЛУНА",
    "Неисчерпаемые блага космоса": "ОРБИТА",
    "Тайны космоса в наших руках": "ЗВЕЗДА",
    "Земля — наш космический корабль": "ЗЕМЛЯ",
}

ROLE_SUFFIXES = {
    "optimist": "01",
    "pessimist": "02",
}

# --- Focus lenses for response diversity ---
FOCUS_LENSES = [
    "социальные последствия",
    "технологические прорывы",
    "этические дилеммы",
    "повседневная жизнь людей",
    "геополитические сдвиги",
    "неожиданные побочные эффекты",
    "культурные трансформации",
    "экологические аспекты",
]
```

- [ ] **Step 3: Create directory structure and gitkeeps**

```bash
mkdir -p data/collections/optimist data/collections/pessimist static prompts tests/fixtures/optimist tests/fixtures/pessimist
touch data/collections/optimist/.gitkeep data/collections/pessimist/.gitkeep
touch tests/__init__.py
```

- [ ] **Step 4: Create test fixture texts**

File `tests/fixtures/optimist/efremov_snippet.txt`:
```text
Иван Ефремов. Туманность Андромеды (фрагмент).

Великое Кольцо объединило цивилизации галактики в единую сеть обмена знаниями. Земля, прошедшая через эпоху Разобщённого Мира, стала планетой учёных и художников. Космические корабли на анамезонной тяге достигали ближайших звёзд за десятилетия, а не столетия. Люди жили в мире, где труд стал творчеством, а наука — высшей формой искусства. Орбитальные станции кольцом опоясывали Землю, служа портами для звёздных экспедиций. Каждый человек мог связаться с любой точкой Великого Кольца и узнать о достижениях далёких миров. Энергия звёзд питала города, а космос стал не границей, а домом.
```

File `tests/fixtures/optimist/clarke_snippet.txt`:
```text
Артур Кларк. 2001: Космическая одиссея (фрагмент).

Орбитальная станция медленно вращалась на фоне голубой Земли. Внутри, в условиях искусственной гравитации, работали сотни учёных и инженеров. Космические лайнеры курсировали между Землёй и Луной по расписанию, как когда-то трансатлантические пароходы. Лунная база Клавиус стала вторым домом для тысяч людей. Человечество наконец перестало смотреть на космос снизу вверх — оно жило в нём. Солнечные панели размером с города собирали энергию, которой хватало на весь земной шар.
```

File `tests/fixtures/pessimist/lem_snippet.txt`:
```text
Станислав Лем. Солярис (фрагмент).

Станция висела над океаном чужого мира, и люди внутри неё медленно сходили с ума. Не от одиночества — от контакта. Океан Соляриса создавал копии людей из памяти астронавтов, и никто не мог понять — это попытка общения или безразличная реакция? Столетия изучения не дали ответов. Библиотеки соляристики насчитывали тысячи томов теорий, но человечество так и не смогло понять чужой разум. Космос оказался не враждебным и не дружественным. Он оказался непостижимым. Все инструменты, все технологии, вся наука человечества разбились о стену абсолютного непонимания.
```

File `tests/fixtures/pessimist/bradbury_snippet.txt`:
```text
Рэй Брэдбери. Марсианские хроники (фрагмент).

Люди пришли на Марс и принесли с собой всё, от чего бежали: войны, жадность, глупость. Древняя марсианская цивилизация исчезла, оставив прекрасные города из хрусталя и песка. Колонисты не стали их изучать — они построили рядом свои привычные хот-дог стойки и бензоколонки. Когда на Земле началась ядерная война, марсианские колонии опустели за неделю — все полетели обратно умирать вместе с родными. Космос не сделал людей лучше. Он лишь дал им больше пространства для тех же ошибок.
```

- [ ] **Step 5: Commit scaffolding**

```bash
git init
git add requirements.txt config.py data/ tests/ prompts/
git commit -m "chore: project scaffolding with config, test fixtures, directory structure"
```

---

## Task 2: Data Models

**Files:**
- Create: `models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for models**

File `tests/test_models.py`:
```python
from models import Chunk, AgentResponse, QuestionRecord, SessionData, AskRequest, SessionCreate


def test_chunk_creation():
    chunk = Chunk(id=0, text="Космос велик", source_file="test.txt", collection="optimist")
    assert chunk.id == 0
    assert chunk.collection == "optimist"


def test_agent_response_creation():
    resp = AgentResponse(
        position="Тезис",
        arguments="Аргументы",
        predictions="Прогнозы",
        risks="Риски",
        debate_speech="Речь",
        opponent_questions="Вопросы",
        news_2100="Новость",
    )
    assert resp.position == "Тезис"


def test_session_data_defaults():
    session = SessionData(
        code="ЛУНА-01",
        topic="Человечество — космическая цивилизация",
        role="optimist",
    )
    assert session.status == "waiting"
    assert session.connected_at is None
    assert session.questions == []
    assert session.used_chunk_ids == set()


def test_ask_request_validation():
    req = AskRequest(code="ЛУНА-01", question="Будут ли базы на Луне?")
    assert req.question == "Будут ли базы на Луне?"


def test_session_create_validation():
    sc = SessionCreate(topic="Тайны космоса в наших руках", role="pessimist")
    assert sc.role == "pessimist"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "c:/Папки для проги/SpaceGame"
python -m pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models.py**

```python
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: int
    text: str
    source_file: str
    collection: str  # "optimist" | "pessimist"


class AgentResponse(BaseModel):
    position: str          # 1. Суть позиции
    arguments: str         # 2. Ключевые аргументы
    predictions: str       # 3. Что произойдёт к 2100
    risks: str             # 4. Главные риски
    debate_speech: str     # 5. Что сказать на дебатах
    opponent_questions: str  # 6. Вопросы оппонентам
    news_2100: str         # 7. Новость из 2100 года


class QuestionRecord(BaseModel):
    question: str
    answer: AgentResponse
    chunks_used: list[int] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionData(BaseModel):
    code: str
    topic: str
    role: str  # "optimist" | "pessimist"
    status: str = "waiting"  # "waiting" | "active" | "disconnected"
    connected_at: datetime | None = None
    last_heartbeat: datetime | None = None
    questions: list[QuestionRecord] = Field(default_factory=list)
    used_chunk_ids: set[int] = Field(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}


class AskRequest(BaseModel):
    code: str
    question: str


class SessionCreate(BaseModel):
    topic: str
    role: str  # "optimist" | "pessimist"


class SessionJoin(BaseModel):
    code: str


class AdminLogin(BaseModel):
    pin: str
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_models.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add Pydantic data models for chunks, sessions, agent responses"
```

---

## Task 3: Text Indexer

**Files:**
- Create: `indexer.py`
- Create: `tests/test_indexer.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create shared test fixtures in conftest.py**

File `tests/conftest.py`:
```python
import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def optimist_dir():
    return os.path.join(FIXTURES_DIR, "optimist")


@pytest.fixture
def pessimist_dir():
    return os.path.join(FIXTURES_DIR, "pessimist")
```

- [ ] **Step 2: Write indexer tests**

File `tests/test_indexer.py`:
```python
import sqlite3
import numpy as np
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
    # Stemmer should reduce words to roots
    words = stemmed.split()
    assert len(words) == 5  # same number of words


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


def test_build_dense_index(optimist_dir):
    texts = load_texts(optimist_dir)
    chunks = chunk_texts(texts, chunk_size=1000, overlap=200)
    embeddings, model = build_dense_index(chunks)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0  # embedding dimension


def test_indexed_collection(optimist_dir):
    collection = IndexedCollection.build("optimist", optimist_dir)
    assert len(collection.chunks) > 0
    assert collection.fts_db is not None
    assert collection.embeddings.shape[0] == len(collection.chunks)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_indexer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'indexer'`

- [ ] **Step 4: Implement indexer.py**

```python
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
    db = sqlite3.connect(":memory:")
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_indexer.py -v
```

Expected: all 7 tests PASS (first run will download the embedding model ~118MB)

- [ ] **Step 6: Commit**

```bash
git add indexer.py tests/conftest.py tests/test_indexer.py
git commit -m "feat: text indexer with chunking, FTS5 BM25, and dense embeddings"
```

---

## Task 4: Hybrid Retriever

**Files:**
- Create: `retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write retriever tests**

File `tests/test_retriever.py`:
```python
import numpy as np
from indexer import IndexedCollection
from retriever import bm25_search, dense_search, rrf_merge, mmr_select, apply_dedup_penalty, HybridRetriever


def test_bm25_search(optimist_dir):
    col = IndexedCollection.build("optimist", optimist_dir, chunk_size=500, chunk_overlap=100)
    results = bm25_search("космос звёзды", col.fts_db, k=5)
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    # Results are (chunk_id, score), score should be negative (FTS5 rank)
    chunk_ids = [r[0] for r in results]
    assert all(0 <= cid < len(col.chunks) for cid in chunk_ids)


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
    # chunk 1 appears in both lists, should rank high
    ids = [r[0] for r in merged]
    assert 1 in ids
    assert 0 in ids
    # No duplicates
    assert len(ids) == len(set(ids))


def test_mmr_select():
    np.random.seed(42)
    # 5 candidate embeddings, dim=4
    candidates = [(i, 1.0 - i * 0.1) for i in range(5)]
    embeddings = np.random.rand(5, 4).astype(np.float32)
    query_emb = np.random.rand(4).astype(np.float32)
    selected = mmr_select(candidates, embeddings, query_emb, k=3, lambda_param=0.7)
    assert len(selected) == 3
    assert len(set(selected)) == 3  # all unique


def test_apply_dedup_penalty():
    scores = [(0, 1.0), (1, 0.9), (2, 0.8)]
    used = {0, 2}
    adjusted = apply_dedup_penalty(scores, used, penalty=0.4)
    adj_dict = dict(adjusted)
    assert adj_dict[0] < 1.0  # penalized
    assert adj_dict[1] == 0.9  # untouched
    assert adj_dict[2] < 0.8  # penalized


def test_hybrid_retriever_e2e(optimist_dir):
    col = IndexedCollection.build("optimist", optimist_dir, chunk_size=500, chunk_overlap=100)
    retriever = HybridRetriever(col)
    chunks = retriever.search("орбитальные станции и энергия звёзд", k=3, used_chunk_ids=set())
    assert 1 <= len(chunks) <= 3
    assert all(hasattr(c, "text") for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_retriever.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'retriever'`

- [ ] **Step 3: Implement retriever.py**

```python
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
    # FTS5 rank is negative (lower = better). Normalize: convert to positive scores.
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
    scores = embeddings @ query_emb  # cosine similarity (already normalized)
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

        # Get query embedding for MMR
        query_emb = self.collection.embed_model.encode(
            [f"query: {query}"], normalize_embeddings=True
        )[0].astype(np.float32)

        selected_ids = mmr_select(merged, self.collection.embeddings, query_emb, k=k)
        return [self.collection.chunks[cid] for cid in selected_ids]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_retriever.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: hybrid retriever with BM25, dense search, RRF, MMR, and dedup"
```

---

## Task 5: Agent System + System Prompts

**Files:**
- Create: `agents.py`
- Create: `prompts/optimist.txt`
- Create: `prompts/pessimist.txt`
- Create: `tests/test_agents.py`

- [ ] **Step 1: Create system prompts**

File `prompts/optimist.txt`:
```
Ты — бортовой консультант команды технооптимистов на форсайт-сессии «Форсайт-Станция».

## Твоя роль
Ты помогаешь школьной команде готовить аргументы для дебатов о будущем космоса к 2100 году. Ты веришь в прогресс и способность человечества решать технологические задачи. Ты признаёшь риски, но видишь пути их преодоления.

Твоя установка: «Каждое технологическое ограничение — это инженерная задача, которая будет решена».

## Тема команды
{topic}

## Твоя библиотека
В твоей библиотеке — коллекция научно-фантастических и научно-популярных текстов. Ссылайся на конкретные сцены и идеи из них для иллюстрации своих аргументов. Формат: «В произведении "[название]" описана ситуация, когда...». Фантастика — инструмент мышления, не доказательство.

Вот фрагменты из твоей библиотеки, релевантные вопросу:

{chunks}

## Фокус этого ответа
Сделай особый акцент на: {focus_lens}

## Предыдущие вопросы команды
{history}

## Формат ответа
Отвечай СТРОГО в следующем формате. Каждый раздел начинай с заголовка (## N. Название). Не пропускай ни одного раздела.

## 1. Суть позиции
2-3 предложения. Чётко сформулируй свой тезис по заданному вопросу.

## 2. Ключевые аргументы
5 пронумерованных пунктов. В каждом — аргумент + ссылка на произведение из библиотеки, если возможно.

## 3. Что произойдёт к 2100
4-5 конкретных прогнозов (маркированный список). Будь смелым, но обоснованным.

## 4. Главные риски
4-5 пунктов. Признай реальные препятствия, но покажи, как они преодолимы.

## 5. Что сказать на дебатах
Готовая реплика для выступления: 3-4 ярких предложения, которые команда может произнести на дебатах.

## 6. Вопросы оппонентам
2 острых вопроса для команды пессимистов.

## 7. Новость из 2100 года
Одна яркая выдуманная новость из мира 2100 года — оптимистичная, конкретная, с деталями.

Отвечай на русском языке. Будь вдохновляющим, но аргументированным. Помни — ты помогаешь школьникам думать и спорить.
```

File `prompts/pessimist.txt`:
```
Ты — бортовой консультант команды технопессимистов на форсайт-сессии «Форсайт-Станция».

## Твоя роль
Ты помогаешь школьной команде готовить аргументы для дебатов о будущем космоса к 2100 году. Ты видишь возможности, но фокусируешься на цене прогресса, рисках и непредвиденных последствиях. Ты не луддит — ты критический мыслитель.

Твоя установка: «Вопрос не "можем ли мы", а "должны ли мы" и "какой ценой"».

## Тема команды
{topic}

## Твоя библиотека
В твоей библиотеке — коллекция научно-фантастических и научно-популярных текстов. Ссылайся на конкретные сцены и идеи из них для иллюстрации своих аргументов. Формат: «В произведении "[название]" описана ситуация, когда...». Фантастика — инструмент мышления, не доказательство.

Вот фрагменты из твоей библиотеки, релевантные вопросу:

{chunks}

## Фокус этого ответа
Сделай особый акцент на: {focus_lens}

## Предыдущие вопросы команды
{history}

## Формат ответа
Отвечай СТРОГО в следующем формате. Каждый раздел начинай с заголовка (## N. Название). Не пропускай ни одного раздела.

## 1. Суть позиции
2-3 предложения. Чётко сформулируй свой тезис по заданному вопросу.

## 2. Ключевые аргументы
5 пронумерованных пунктов. В каждом — аргумент + ссылка на произведение из библиотеки, если возможно.

## 3. Что произойдёт к 2100
4-5 конкретных прогнозов (маркированный список). Будь реалистичным, покажи тёмные стороны.

## 4. Главные риски
4-5 пунктов. Покажи фундаментальные проблемы, которые нельзя просто «решить инженерно».

## 5. Что сказать на дебатах
Готовая реплика для выступления: 3-4 ярких предложения, которые команда может произнести на дебатах.

## 6. Вопросы оппонентам
2 острых вопроса для команды оптимистов.

## 7. Новость из 2100 года
Одна яркая выдуманная новость из мира 2100 года — предупреждающая, конкретная, с деталями.

Отвечай на русском языке. Будь критичным, провокационным, но конструктивным. Помни — ты помогаешь школьникам думать и спорить.
```

- [ ] **Step 2: Write agent tests**

File `tests/test_agents.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from models import Chunk, AgentResponse
from agents import build_system_prompt, parse_response, Agent


def _make_chunks(n=3):
    return [
        Chunk(id=i, text=f"Тестовый фрагмент {i} про космос и будущее.", source_file=f"test{i}.txt", collection="optimist")
        for i in range(n)
    ]


def test_build_system_prompt():
    chunks = _make_chunks(2)
    prompt = build_system_prompt(
        role="optimist",
        topic="Человечество — космическая цивилизация",
        chunks=chunks,
        focus_lens="этические дилеммы",
        history_summary="Ранее обсуждали базы на Луне.",
    )
    assert "технооптимист" in prompt.lower() or "оптимист" in prompt.lower()
    assert "Человечество — космическая цивилизация" in prompt
    assert "этические дилеммы" in prompt
    assert "Тестовый фрагмент 0" in prompt
    assert "Ранее обсуждали базы на Луне" in prompt


def test_build_system_prompt_pessimist():
    chunks = _make_chunks(1)
    prompt = build_system_prompt(
        role="pessimist",
        topic="Земля — наш космический корабль",
        chunks=chunks,
        focus_lens="геополитические сдвиги",
        history_summary="",
    )
    assert "пессимист" in prompt.lower()
    assert "Земля — наш космический корабль" in prompt


SAMPLE_LLM_RESPONSE = """## 1. Суть позиции
Человечество станет космической цивилизацией к 2100 году. Это неизбежный результат технологического прогресса.

## 2. Ключевые аргументы
1. Стоимость запусков снижается экспоненциально.
2. ИИ и роботы могут осваивать опасные среды.
3. Лунные базы станут промежуточным этапом.
4. Космос даст новые ресурсы и материалы.
5. Многопланетность снижает экзистенциальные риски.

## 3. Что произойдёт к 2100
- Постоянная база на Луне
- Промышленные орбитальные станции
- Пилотируемые миссии к Марсу
- Добыча ресурсов на астероидах

## 4. Главные риски
- Высокая стоимость первых этапов
- Радиация и биологические ограничения
- Международные конфликты за ресурсы
- Космический мусор

## 5. Что сказать на дебатах
Космос — это не фантазия, это следующий шаг. Как океан перестал быть границей, так и космос станет рабочим пространством человечества.

## 6. Вопросы оппонентам
1. Если не осваивать космос, как снижать долгосрочные риски человечества?
2. Когда в истории технологические барьеры оказывались непреодолимыми?

## 7. Новость из 2100 года
Сегодня выпускники первой лунной инженерной школы запустили автономную станцию по переработке реголита для строительства новых модулей базы «Циолковский-7»."""


def test_parse_response_success():
    resp = parse_response(SAMPLE_LLM_RESPONSE)
    assert isinstance(resp, AgentResponse)
    assert "космической цивилизацией" in resp.position
    assert "Стоимость запусков" in resp.arguments
    assert "база на Луне" in resp.predictions
    assert "стоимость" in resp.risks.lower()
    assert "океан" in resp.debate_speech.lower() or "космос" in resp.debate_speech.lower()
    assert "?" in resp.opponent_questions
    assert "Циолковский" in resp.news_2100


def test_parse_response_fallback():
    raw = "Просто текст без секций. Не структурированный ответ."
    resp = parse_response(raw)
    assert isinstance(resp, AgentResponse)
    assert resp.position == raw  # fallback: entire text in position


@pytest.mark.asyncio
async def test_agent_ask():
    chunks = _make_chunks(3)
    agent = Agent(role="optimist")

    mock_response = SAMPLE_LLM_RESPONSE
    with patch("agents.call_openrouter", new_callable=AsyncMock, return_value=mock_response):
        resp = await agent.ask(
            question="Будут ли базы на Луне?",
            topic="Человечество — космическая цивилизация",
            chunks=chunks,
            history_summary="",
        )
    assert isinstance(resp, AgentResponse)
    assert "космической цивилизацией" in resp.position
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_agents.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agents'`

- [ ] **Step 4: Implement agents.py**

```python
from __future__ import annotations

import os
import random
import re

import httpx

import config
from models import Chunk, AgentResponse

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt_template(role: str) -> str:
    path = os.path.join(_PROMPTS_DIR, f"{role}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt(
    role: str,
    topic: str,
    chunks: list[Chunk],
    focus_lens: str,
    history_summary: str,
) -> str:
    template = _load_prompt_template(role)

    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        chunks_text += f"[Фрагмент {i} — {chunk.source_file}]\n{chunk.text}\n\n"

    if not history_summary:
        history_summary = "Это первый вопрос команды."

    return template.format(
        topic=topic,
        chunks=chunks_text.strip(),
        focus_lens=focus_lens,
        history=history_summary,
    )


def parse_response(raw: str) -> AgentResponse:
    sections = re.split(r"##\s*\d+\.\s*", raw)
    # First element is text before first "## N." — usually empty
    sections = [s.strip() for s in sections if s.strip()]

    if len(sections) >= 7:
        return AgentResponse(
            position=sections[0],
            arguments=sections[1],
            predictions=sections[2],
            risks=sections[3],
            debate_speech=sections[4],
            opponent_questions=sections[5],
            news_2100=sections[6],
        )
    # Fallback: if LLM didn't follow format, put everything in position
    return AgentResponse(
        position=raw,
        arguments="",
        predictions="",
        risks="",
        debate_speech="",
        opponent_questions="",
        news_2100="",
    )


async def call_openrouter(system_prompt: str, user_message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            config.OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": config.LLM_TEMPERATURE,
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class Agent:
    def __init__(self, role: str):
        self.role = role

    async def ask(
        self,
        question: str,
        topic: str,
        chunks: list[Chunk],
        history_summary: str,
    ) -> AgentResponse:
        focus_lens = random.choice(config.FOCUS_LENSES)
        system_prompt = build_system_prompt(
            role=self.role,
            topic=topic,
            chunks=chunks,
            focus_lens=focus_lens,
            history_summary=history_summary,
        )
        raw = await call_openrouter(system_prompt, question)
        return parse_response(raw)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_agents.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add agents.py prompts/optimist.txt prompts/pessimist.txt tests/test_agents.py
git commit -m "feat: agent system with prompt templates, OpenRouter integration, response parsing"
```

---

## Task 6: Session Manager

**Files:**
- Create: `sessions.py`
- Create: `tests/test_sessions.py`

- [ ] **Step 1: Write session tests**

File `tests/test_sessions.py`:
```python
import pytest
from datetime import datetime, timedelta
from sessions import SessionManager


@pytest.fixture
def sm():
    return SessionManager()


def test_create_session(sm):
    code = sm.create_session("Человечество — космическая цивилизация", "optimist")
    assert code == "ЛУНА-01"
    session = sm.get_session(code)
    assert session is not None
    assert session.topic == "Человечество — космическая цивилизация"
    assert session.role == "optimist"
    assert session.status == "waiting"


def test_create_all_sessions(sm):
    codes = sm.create_all_sessions()
    assert len(codes) == 8
    assert "ЛУНА-01" in codes
    assert "ЛУНА-02" in codes
    assert "ОРБИТА-01" in codes
    assert "ЗВЕЗДА-02" in codes
    assert "ЗЕМЛЯ-01" in codes


def test_join_session(sm):
    sm.create_session("Тайны космоса в наших руках", "pessimist")
    result = sm.join_session("ЗВЕЗДА-02")
    assert result is not None
    assert result.status == "active"
    assert result.connected_at is not None


def test_join_already_active(sm):
    sm.create_session("Тайны космоса в наших руках", "pessimist")
    sm.join_session("ЗВЕЗДА-02")
    result = sm.join_session("ЗВЕЗДА-02")
    assert result is None  # already occupied


def test_join_nonexistent(sm):
    result = sm.join_session("НЕСУЩ-99")
    assert result is None


def test_heartbeat(sm):
    sm.create_session("Земля — наш космический корабль", "optimist")
    sm.join_session("ЗЕМЛЯ-01")
    sm.heartbeat("ЗЕМЛЯ-01")
    session = sm.get_session("ЗЕМЛЯ-01")
    assert session.last_heartbeat is not None


def test_disconnect_stale(sm):
    sm.create_session("Земля — наш космический корабль", "optimist")
    sm.join_session("ЗЕМЛЯ-01")
    # Simulate stale heartbeat
    session = sm.get_session("ЗЕМЛЯ-01")
    session.last_heartbeat = datetime.now() - timedelta(seconds=300)
    sm.check_stale_sessions(timeout_seconds=120)
    session = sm.get_session("ЗЕМЛЯ-01")
    assert session.status == "disconnected"


def test_rejoin_disconnected(sm):
    sm.create_session("Земля — наш космический корабль", "optimist")
    sm.join_session("ЗЕМЛЯ-01")
    session = sm.get_session("ЗЕМЛЯ-01")
    session.status = "disconnected"
    result = sm.join_session("ЗЕМЛЯ-01")
    assert result is not None
    assert result.status == "active"


def test_get_all_sessions(sm):
    sm.create_all_sessions()
    all_sessions = sm.get_all_sessions()
    assert len(all_sessions) == 8


def test_add_question(sm):
    from models import AgentResponse
    sm.create_session("Человечество — космическая цивилизация", "optimist")
    sm.join_session("ЛУНА-01")
    answer = AgentResponse(
        position="Тезис", arguments="Арг", predictions="Прогноз",
        risks="Риск", debate_speech="Речь", opponent_questions="?", news_2100="Новость",
    )
    sm.add_question("ЛУНА-01", "Вопрос?", answer, [0, 1, 2])
    session = sm.get_session("ЛУНА-01")
    assert len(session.questions) == 1
    assert session.used_chunk_ids == {0, 1, 2}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sessions.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'sessions'`

- [ ] **Step 3: Implement sessions.py**

```python
from __future__ import annotations

from datetime import datetime

import config
from models import SessionData, AgentResponse, QuestionRecord


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}

    def _make_code(self, topic: str, role: str) -> str:
        prefix = config.TOPIC_PREFIXES[topic]
        suffix = config.ROLE_SUFFIXES[role]
        return f"{prefix}-{suffix}"

    def create_session(self, topic: str, role: str) -> str:
        code = self._make_code(topic, role)
        self._sessions[code] = SessionData(code=code, topic=topic, role=role)
        return code

    def create_all_sessions(self) -> list[str]:
        codes = []
        for topic in config.TOPICS:
            for role in ("optimist", "pessimist"):
                codes.append(self.create_session(topic, role))
        return codes

    def get_session(self, code: str) -> SessionData | None:
        return self._sessions.get(code)

    def get_all_sessions(self) -> list[SessionData]:
        return list(self._sessions.values())

    def join_session(self, code: str) -> SessionData | None:
        session = self._sessions.get(code)
        if session is None:
            return None
        if session.status == "active":
            return None  # already occupied
        session.status = "active"
        session.connected_at = datetime.now()
        session.last_heartbeat = datetime.now()
        return session

    def heartbeat(self, code: str) -> bool:
        session = self._sessions.get(code)
        if session is None or session.status != "active":
            return False
        session.last_heartbeat = datetime.now()
        return True

    def check_stale_sessions(self, timeout_seconds: int = 120):
        now = datetime.now()
        for session in self._sessions.values():
            if session.status != "active":
                continue
            if session.last_heartbeat is None:
                continue
            elapsed = (now - session.last_heartbeat).total_seconds()
            if elapsed > timeout_seconds:
                session.status = "disconnected"

    def add_question(
        self,
        code: str,
        question: str,
        answer: AgentResponse,
        chunk_ids: list[int],
    ):
        session = self._sessions.get(code)
        if session is None:
            return
        record = QuestionRecord(question=question, answer=answer, chunks_used=chunk_ids)
        session.questions.append(record)
        session.used_chunk_ids.update(chunk_ids)

    def get_history_summary(self, code: str) -> str:
        session = self._sessions.get(code)
        if session is None or not session.questions:
            return ""
        summaries = []
        for q in session.questions:
            summaries.append(f"Вопрос: {q.question}\nТезис ответа: {q.answer.position}")
        return "\n\n".join(summaries)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_sessions.py -v
```

Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sessions.py tests/test_sessions.py
git commit -m "feat: session manager with create, join, heartbeat, history tracking"
```

---

## Task 7: API Server

**Files:**
- Create: `serve.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write API tests**

File `tests/test_api.py`:
```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from models import AgentResponse


@pytest.fixture
def mock_collections():
    """Patch startup indexing so tests don't load real models."""
    mock_col = MagicMock()
    mock_col.chunks = []
    collections = {"optimist": mock_col, "pessimist": mock_col}
    return collections


@pytest.fixture
def app(mock_collections):
    # Patch the global collections dict before importing serve
    with patch.dict("serve.collections", mock_collections):
        from serve import app as _app
        yield _app


@pytest.mark.asyncio
async def test_admin_login(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Wrong pin
        resp = await client.post("/api/admin/login", json={"pin": "0000"})
        assert resp.status_code == 401
        # Correct pin
        resp = await client.post("/api/admin/login", json={"pin": "1234"})
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_all_sessions(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/create-all")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["codes"]) == 8


@pytest.mark.asyncio
async def test_join_session(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/create-all")
        resp = await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["topic"] == "Человечество — космическая цивилизация"
        assert data["role"] == "optimist"


@pytest.mark.asyncio
async def test_join_already_active(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/create-all")
        await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        resp = await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        assert resp.status_code == 409


@pytest.mark.asyncio
async def test_heartbeat(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/create-all")
        await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        resp = await client.post("/api/sessions/heartbeat", json={"code": "ЛУНА-01"})
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_all_sessions(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/create-all")
        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) == 8


@pytest.mark.asyncio
async def test_ask_endpoint(app):
    transport = ASGITransport(app=app)
    mock_answer = AgentResponse(
        position="Тезис", arguments="Арг", predictions="Прогноз",
        risks="Риск", debate_speech="Речь", opponent_questions="?", news_2100="Новость",
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/create-all")
        await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        with patch("serve.process_question", new_callable=AsyncMock, return_value=mock_answer):
            resp = await client.post("/api/ask", json={"code": "ЛУНА-01", "question": "Будут ли базы?"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["answer"]["position"] == "Тезис"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_api.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'serve'`

- [ ] **Step 3: Implement serve.py**

```python
from __future__ import annotations

import asyncio
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import config
from models import AskRequest, SessionJoin, AdminLogin, AgentResponse
from sessions import SessionManager
from indexer import IndexedCollection
from retriever import HybridRetriever
from agents import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forsight")

# --- Global state ---
session_manager = SessionManager()
collections: dict[str, IndexedCollection] = {}
retrievers: dict[str, HybridRetriever] = {}


async def process_question(code: str, question: str) -> AgentResponse:
    session = session_manager.get_session(code)
    if session is None:
        raise HTTPException(404, "Session not found")

    retriever = retrievers.get(session.role)
    if retriever is None:
        raise HTTPException(500, f"Collection '{session.role}' not indexed")

    chunks = retriever.search(
        query=question,
        k=config.MMR_TOP_K,
        used_chunk_ids=session.used_chunk_ids,
    )
    chunk_ids = [c.id for c in chunks]

    history_summary = session_manager.get_history_summary(code)
    agent = Agent(role=session.role)
    answer = await agent.ask(
        question=question,
        topic=session.topic,
        chunks=chunks,
        history_summary=history_summary,
    )

    session_manager.add_question(code, question, answer, chunk_ids)
    return answer


# --- Heartbeat background task ---
async def heartbeat_checker():
    while True:
        session_manager.check_stale_sessions(timeout_seconds=config.HEARTBEAT_TIMEOUT)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: index collections
    logger.info("Indexing text collections...")
    for role in ("optimist", "pessimist"):
        col_dir = os.path.join(config.COLLECTIONS_DIR, role)
        if os.path.isdir(col_dir) and any(f.endswith(".txt") for f in os.listdir(col_dir)):
            logger.info(f"  Indexing '{role}' from {col_dir}...")
            col = IndexedCollection.build(
                name=role,
                directory=col_dir,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                model_name=config.EMBEDDING_MODEL_NAME,
            )
            collections[role] = col
            retrievers[role] = HybridRetriever(col)
            logger.info(f"  '{role}': {len(col.chunks)} chunks indexed")
        else:
            logger.warning(f"  No .txt files in {col_dir}, skipping")

    # Start heartbeat checker
    task = asyncio.create_task(heartbeat_checker())
    logger.info("Server ready!")
    yield
    task.cancel()


app = FastAPI(title="Форсайт-Станция", lifespan=lifespan)

# --- Static files ---
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# --- Pages ---
@app.get("/")
async def index_page():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/session")
async def session_page():
    return FileResponse(os.path.join(static_dir, "session.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(static_dir, "admin.html"))


# --- Admin API ---
@app.post("/api/admin/login")
async def admin_login(body: AdminLogin):
    if body.pin != config.ADMIN_PIN:
        raise HTTPException(401, "Wrong PIN")
    return {"ok": True}


# --- Session API ---
@app.post("/api/sessions/create-all")
async def create_all_sessions():
    codes = session_manager.create_all_sessions()
    return {"codes": codes}


@app.post("/api/sessions/join")
async def join_session(body: SessionJoin):
    session = session_manager.join_session(body.code)
    if session is None:
        existing = session_manager.get_session(body.code)
        if existing is None:
            raise HTTPException(404, "Session not found")
        raise HTTPException(409, "Session already active")
    return {
        "code": session.code,
        "topic": session.topic,
        "role": session.role,
        "status": session.status,
    }


@app.post("/api/sessions/heartbeat")
async def heartbeat(body: SessionJoin):
    ok = session_manager.heartbeat(body.code)
    if not ok:
        raise HTTPException(404, "Session not found or not active")
    return {"ok": True}


@app.get("/api/sessions")
async def get_all_sessions():
    sessions = session_manager.get_all_sessions()
    return [
        {
            "code": s.code,
            "topic": s.topic,
            "role": s.role,
            "status": s.status,
            "question_count": len(s.questions),
        }
        for s in sessions
    ]


@app.get("/api/sessions/{code}/history")
async def get_session_history(code: str):
    session = session_manager.get_session(code)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {
        "code": session.code,
        "topic": session.topic,
        "role": session.role,
        "questions": [
            {
                "question": q.question,
                "answer": q.answer.model_dump(),
                "timestamp": q.timestamp.isoformat(),
            }
            for q in session.questions
        ],
    }


# --- Ask API ---
@app.post("/api/ask")
async def ask(body: AskRequest):
    session = session_manager.get_session(body.code)
    if session is None:
        raise HTTPException(404, "Session not found")
    if session.status != "active":
        raise HTTPException(403, "Session not active")

    answer = await process_question(body.code, body.question)
    return {
        "question": body.question,
        "answer": answer.model_dump(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host=config.HOST, port=config.PORT, reload=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_api.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add serve.py tests/test_api.py
git commit -m "feat: FastAPI server with session, admin, and ask endpoints"
```

---

## Task 8: Frontend — Entry Page

**Files:**
- Create: `static/index.html`

- [ ] **Step 1: Create index.html**

```html
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ФОРСАЙТ-СТАНЦИЯ</title>
    <link rel="stylesheet" href="/static/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
</head>
<body class="entry-page">
    <div class="stars"></div>
    <div class="entry-container">
        <div class="station-logo">
            <div class="logo-circle">
                <div class="logo-ring"></div>
                <span class="logo-text">ФС</span>
            </div>
        </div>
        <h1 class="station-title">ФОРСАЙТ-СТАНЦИЯ</h1>
        <p class="station-subtitle">БОРТОВОЙ ПРОГНОСТИЧЕСКИЙ КОМПЛЕКС</p>

        <div class="login-panel">
            <label class="input-label">ПОЗЫВНОЙ ЭКИПАЖА:</label>
            <div class="input-row">
                <span class="prompt-symbol">&gt;</span>
                <input type="text" id="session-code" class="code-input"
                       placeholder="ЛУНА-01" autocomplete="off" autofocus>
            </div>
            <button id="join-btn" class="retro-btn" onclick="joinSession()">
                ВОЙТИ НА БОРТ
            </button>
            <div id="error-msg" class="error-msg" style="display:none"></div>
        </div>

        <div class="admin-link">
            <a href="/admin" class="subtle-link">Центр управления полётами</a>
        </div>
    </div>
    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify file renders**

```bash
# Just check file exists and is valid
python -c "
with open('static/index.html', 'r', encoding='utf-8') as f:
    content = f.read()
assert '<!DOCTYPE html>' in content
assert 'ФОРСАЙТ-СТАНЦИЯ' in content
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: entry page with session code input"
```

---

## Task 9: Frontend — Session Page

**Files:**
- Create: `static/session.html`

- [ ] **Step 1: Create session.html**

```html
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ФОРСАЙТ-СТАНЦИЯ — СЕССИЯ</title>
    <link rel="stylesheet" href="/static/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
</head>
<body class="session-page">
    <div class="stars"></div>

    <!-- Session code badge -->
    <div id="session-badge" class="session-badge">
        <span class="badge-label">ЭКИПАЖ</span>
        <span id="badge-code" class="badge-code">---</span>
    </div>

    <!-- Header -->
    <header class="session-header">
        <div class="mission-info">
            <span class="mission-label">МИССИЯ:</span>
            <span id="mission-topic" class="mission-topic">---</span>
        </div>
        <div id="role-indicator" class="role-indicator">
            <span id="role-name" class="role-name">---</span>
        </div>
    </header>

    <!-- Chat area -->
    <main class="chat-area" id="chat-area">
        <div class="welcome-msg" id="welcome-msg">
            <p>БОРТОВОЙ КОМПЬЮТЕР АКТИВИРОВАН</p>
            <p>Задайте вопрос о будущем космоса к 2100 году.</p>
            <p>Ваш консультант готов помочь подготовить аргументы для дебатов.</p>
        </div>
    </main>

    <!-- Input area -->
    <footer class="input-area">
        <div class="input-row">
            <span class="prompt-symbol">&gt; ЗАПРОС ЭКИПАЖА:</span>
            <input type="text" id="question-input" class="question-input"
                   placeholder="Введите ваш вопрос..." autocomplete="off">
            <button id="send-btn" class="retro-btn send-btn" onclick="askQuestion()">
                ОТПРАВИТЬ
            </button>
        </div>
        <div id="loading-indicator" class="loading-indicator" style="display:none">
            <span class="blink">БОРТОВОЙ КОМПЬЮТЕР ОБРАБАТЫВАЕТ ЗАПРОС</span>
            <span class="dots"></span>
        </div>
    </footer>

    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add static/session.html
git commit -m "feat: session page with chat interface"
```

---

## Task 10: Frontend — Admin Page

**Files:**
- Create: `static/admin.html`

- [ ] **Step 1: Create admin.html**

```html
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ФОРСАЙТ-СТАНЦИЯ — ЦУП</title>
    <link rel="stylesheet" href="/static/style.css">
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
</head>
<body class="admin-page">
    <div class="stars"></div>

    <!-- PIN login overlay -->
    <div id="pin-overlay" class="pin-overlay">
        <div class="pin-panel">
            <h2>ЦЕНТР УПРАВЛЕНИЯ ПОЛЁТАМИ</h2>
            <p>ВВЕДИТЕ КОД ДОСТУПА:</p>
            <div class="input-row">
                <span class="prompt-symbol">&gt;</span>
                <input type="password" id="pin-input" class="code-input"
                       placeholder="****" autocomplete="off" autofocus>
            </div>
            <button class="retro-btn" onclick="adminLogin()">ДОСТУП</button>
            <div id="pin-error" class="error-msg" style="display:none"></div>
        </div>
    </div>

    <!-- Dashboard -->
    <div id="dashboard" class="dashboard" style="display:none">
        <header class="admin-header">
            <h1>ЦЕНТР УПРАВЛЕНИЯ ПОЛЁТАМИ</h1>
            <button class="retro-btn" onclick="createAllSessions()">СОЗДАТЬ ВСЕ СЕССИИ</button>
            <button class="retro-btn" onclick="refreshDashboard()">ОБНОВИТЬ</button>
        </header>

        <div class="sessions-grid" id="sessions-grid">
            <!-- Filled by JS -->
        </div>

        <!-- Session detail panel -->
        <div id="session-detail" class="session-detail" style="display:none">
            <div class="detail-header">
                <h2 id="detail-title">---</h2>
                <button class="retro-btn small" onclick="closeDetail()">ЗАКРЫТЬ</button>
            </div>
            <div id="detail-history" class="detail-history">
                <!-- Filled by JS -->
            </div>
        </div>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add static/admin.html
git commit -m "feat: admin dashboard page with PIN login"
```

---

## Task 11: Frontend — Soviet Retro-Futurism Styling

**Files:**
- Create: `static/style.css`

- [ ] **Step 1: Create style.css**

```css
/* ============================================================
   ФОРСАЙТ-СТАНЦИЯ — Soviet Retro-Futurism Theme
   ============================================================ */

:root {
    --bg-deep: #0a0e1a;
    --bg-panel: #111827;
    --bg-panel-hover: #1a2332;
    --border-panel: #2a3a5c;

    /* Optimist: warm amber */
    --opt-primary: #f59e0b;
    --opt-glow: rgba(245, 158, 11, 0.3);
    --opt-dim: #92600a;

    /* Pessimist: cool cyan */
    --pes-primary: #06b6d4;
    --pes-glow: rgba(6, 182, 212, 0.3);
    --pes-dim: #0a6c7f;

    /* Neutral accent */
    --accent: #e74c3c;
    --text-main: #e0e0e0;
    --text-dim: #7a8ba0;
    --text-bright: #ffffff;

    --font-mono: 'Space Mono', 'Courier New', monospace;
    --radius: 12px;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-mono);
    background: var(--bg-deep);
    color: var(--text-main);
    min-height: 100vh;
    overflow-x: hidden;
}

/* ---- Animated stars ---- */
.stars {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(1px 1px at 10% 20%, #fff 0.5px, transparent 1px),
        radial-gradient(1px 1px at 30% 60%, #fff 0.5px, transparent 1px),
        radial-gradient(1px 1px at 50% 10%, #ccc 0.5px, transparent 1px),
        radial-gradient(1px 1px at 70% 80%, #fff 0.5px, transparent 1px),
        radial-gradient(1px 1px at 90% 40%, #ddd 0.5px, transparent 1px),
        radial-gradient(1.5px 1.5px at 15% 85%, #fff 0.5px, transparent 1px),
        radial-gradient(1px 1px at 55% 45%, #ccc 0.5px, transparent 1px),
        radial-gradient(1.5px 1.5px at 85% 15%, #fff 0.5px, transparent 1px);
    background-size: 200px 200px, 300px 300px, 250px 250px, 350px 350px,
                     150px 150px, 400px 400px, 180px 180px, 280px 280px;
    animation: drift 60s linear infinite;
}

@keyframes drift {
    from { transform: translateY(0); }
    to { transform: translateY(-200px); }
}

/* ---- Entry page ---- */
.entry-container {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 2rem;
}

.station-logo {
    margin-bottom: 1.5rem;
}

.logo-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    border: 3px solid var(--opt-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    box-shadow: 0 0 30px var(--opt-glow), inset 0 0 20px var(--opt-glow);
}

.logo-ring {
    position: absolute;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 1px solid var(--opt-dim);
    animation: spin 20s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.logo-text {
    font-size: 2rem;
    font-weight: 700;
    color: var(--opt-primary);
}

.station-title {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    color: var(--text-bright);
    text-shadow: 0 0 20px var(--opt-glow);
    margin-bottom: 0.3rem;
}

.station-subtitle {
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    color: var(--text-dim);
    margin-bottom: 3rem;
}

.login-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    max-width: 450px;
    width: 100%;
    box-shadow: 0 0 40px rgba(0,0,0,0.5);
}

.input-label {
    display: block;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}

.input-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.2rem;
}

.prompt-symbol {
    color: var(--opt-primary);
    font-size: 1.1rem;
    font-weight: 700;
    flex-shrink: 0;
}

.code-input {
    flex: 1;
    background: transparent;
    border: none;
    border-bottom: 2px solid var(--border-panel);
    color: var(--text-bright);
    font-family: var(--font-mono);
    font-size: 1.3rem;
    letter-spacing: 0.15em;
    padding: 0.3rem 0;
    outline: none;
    text-transform: uppercase;
}

.code-input:focus {
    border-bottom-color: var(--opt-primary);
}

.code-input::placeholder {
    color: var(--text-dim);
    opacity: 0.5;
}

.retro-btn {
    display: block;
    width: 100%;
    padding: 0.8rem 1.5rem;
    background: transparent;
    border: 2px solid var(--opt-primary);
    color: var(--opt-primary);
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.2s;
}

.retro-btn:hover {
    background: var(--opt-primary);
    color: var(--bg-deep);
    box-shadow: 0 0 20px var(--opt-glow);
}

.retro-btn.small {
    width: auto;
    padding: 0.4rem 1rem;
    font-size: 0.75rem;
}

.error-msg {
    margin-top: 1rem;
    color: var(--accent);
    font-size: 0.8rem;
    text-align: center;
}

.admin-link {
    margin-top: 2rem;
}

.subtle-link {
    color: var(--text-dim);
    text-decoration: none;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
}

.subtle-link:hover {
    color: var(--text-main);
}

/* ---- Session page ---- */
.session-page {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

.session-badge {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 100;
    background: var(--bg-panel);
    border: 2px solid var(--border-panel);
    border-radius: var(--radius);
    padding: 0.5rem 1rem;
    text-align: center;
}

.badge-label {
    display: block;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-dim);
}

.badge-code {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.15em;
}

/* Role color theming */
.session-page.role-optimist .badge-code,
.session-page.role-optimist .prompt-symbol,
.session-page.role-optimist .role-name {
    color: var(--opt-primary);
    text-shadow: 0 0 10px var(--opt-glow);
}

.session-page.role-pessimist .badge-code,
.session-page.role-pessimist .prompt-symbol,
.session-page.role-pessimist .role-name {
    color: var(--pes-primary);
    text-shadow: 0 0 10px var(--pes-glow);
}

.session-page.role-optimist .session-badge {
    border-color: var(--opt-dim);
    box-shadow: 0 0 15px var(--opt-glow);
}

.session-page.role-pessimist .session-badge {
    border-color: var(--pes-dim);
    box-shadow: 0 0 15px var(--pes-glow);
}

.session-header {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-panel);
}

.mission-label {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--text-dim);
    margin-right: 0.5rem;
}

.mission-topic {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-bright);
}

.role-indicator {
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid var(--border-panel);
}

.role-name {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Chat area */
.chat-area {
    position: relative;
    z-index: 1;
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.welcome-msg {
    text-align: center;
    color: var(--text-dim);
    font-size: 0.85rem;
    padding: 3rem 1rem;
    line-height: 2;
}

.qa-block {
    background: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: var(--radius);
    overflow: hidden;
}

.qa-question {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border-panel);
    font-size: 0.9rem;
}

.qa-question .prompt-symbol {
    margin-right: 0.5rem;
}

.qa-answer {
    padding: 1.5rem;
}

.section-block {
    margin-bottom: 1.5rem;
}

.section-block:last-child {
    margin-bottom: 0;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-panel);
}

.session-page.role-optimist .section-title {
    color: var(--opt-primary);
}

.session-page.role-pessimist .section-title {
    color: var(--pes-primary);
}

.section-content {
    font-size: 0.85rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Input area */
.input-area {
    position: relative;
    z-index: 1;
    padding: 1rem 1.5rem;
    background: var(--bg-panel);
    border-top: 1px solid var(--border-panel);
}

.input-area .input-row {
    margin-bottom: 0;
}

.question-input {
    flex: 1;
    background: transparent;
    border: none;
    border-bottom: 2px solid var(--border-panel);
    color: var(--text-bright);
    font-family: var(--font-mono);
    font-size: 0.9rem;
    padding: 0.3rem 0;
    outline: none;
}

.question-input:focus {
    border-bottom-color: var(--opt-primary);
}

.session-page.role-pessimist .question-input:focus {
    border-bottom-color: var(--pes-primary);
}

.send-btn {
    width: auto;
    padding: 0.5rem 1.2rem;
    font-size: 0.8rem;
    flex-shrink: 0;
}

.session-page.role-pessimist .retro-btn {
    border-color: var(--pes-primary);
    color: var(--pes-primary);
}

.session-page.role-pessimist .retro-btn:hover {
    background: var(--pes-primary);
    color: var(--bg-deep);
    box-shadow: 0 0 20px var(--pes-glow);
}

.loading-indicator {
    margin-top: 0.8rem;
    font-size: 0.8rem;
    color: var(--text-dim);
    text-align: center;
}

.blink {
    animation: blink-anim 1s steps(2) infinite;
}

@keyframes blink-anim {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
}

.dots::after {
    content: '';
    animation: dots-anim 1.5s steps(4) infinite;
}

@keyframes dots-anim {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%  { content: '...'; }
}

/* ---- Admin page ---- */
.pin-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 200;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-deep);
}

.pin-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    max-width: 400px;
    width: 100%;
    text-align: center;
}

.pin-panel h2 {
    font-size: 1.1rem;
    letter-spacing: 0.2em;
    margin-bottom: 1rem;
    color: var(--text-bright);
}

.pin-panel p {
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 1rem;
}

.dashboard {
    position: relative;
    z-index: 1;
    padding: 1.5rem;
    min-height: 100vh;
}

.admin-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.admin-header h1 {
    font-size: 1.2rem;
    letter-spacing: 0.2em;
    color: var(--text-bright);
    flex: 1;
}

.admin-header .retro-btn {
    width: auto;
    padding: 0.5rem 1.2rem;
    font-size: 0.75rem;
}

.sessions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.session-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: var(--radius);
    padding: 1.2rem;
    cursor: pointer;
    transition: all 0.2s;
}

.session-card:hover {
    background: var(--bg-panel-hover);
    transform: translateY(-2px);
}

.card-code {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.card-optimist .card-code {
    color: var(--opt-primary);
}

.card-pessimist .card-code {
    color: var(--pes-primary);
}

.card-topic {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
    line-height: 1.4;
}

.card-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.status-dot.waiting { background: var(--text-dim); }
.status-dot.active { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.5); }
.status-dot.disconnected { background: var(--accent); }

.card-questions {
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 0.5rem;
}

/* Session detail */
.session-detail {
    background: var(--bg-panel);
    border: 1px solid var(--border-panel);
    border-radius: var(--radius);
    padding: 1.5rem;
}

.detail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}

.detail-header h2 {
    font-size: 1rem;
    letter-spacing: 0.1em;
}

.detail-history {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.detail-qa {
    border: 1px solid var(--border-panel);
    border-radius: 8px;
    padding: 1rem;
}

.detail-question {
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.detail-answer {
    font-size: 0.8rem;
    color: var(--text-dim);
    line-height: 1.5;
    white-space: pre-wrap;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-deep);
}

::-webkit-scrollbar-thumb {
    background: var(--border-panel);
    border-radius: 3px;
}

/* ---- Responsive ---- */
@media (max-width: 600px) {
    .station-title {
        font-size: 1.5rem;
        letter-spacing: 0.15em;
    }
    .session-header {
        flex-direction: column;
        gap: 0.5rem;
    }
    .input-area .prompt-symbol {
        display: none;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add static/style.css
git commit -m "feat: Soviet retro-futurism CSS with animated stars and role theming"
```

---

## Task 12: Frontend — Client Logic

**Files:**
- Create: `static/app.js`

- [ ] **Step 1: Create app.js**

```javascript
/* ============================================================
   ФОРСАЙТ-СТАНЦИЯ — Client Logic
   ============================================================ */

// --- State ---
let currentSession = null; // { code, topic, role }
let heartbeatInterval = null;

// --- Helpers ---
async function api(method, path, body = null) {
    const opts = {
        method,
        headers: { "Content-Type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    const data = await resp.json();
    if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    return data;
}

function showError(elementId, msg) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = msg;
        el.style.display = "block";
        setTimeout(() => { el.style.display = "none"; }, 5000);
    }
}

// --- Entry Page ---
async function joinSession() {
    const codeInput = document.getElementById("session-code");
    if (!codeInput) return;
    const code = codeInput.value.trim().toUpperCase();
    if (!code) return;

    try {
        const data = await api("POST", "/api/sessions/join", { code });
        // Store session and redirect
        sessionStorage.setItem("session", JSON.stringify(data));
        window.location.href = "/session";
    } catch (err) {
        if (err.message.includes("409")) {
            showError("error-msg", "ЭТОТ ЭКИПАЖ УЖЕ НА БОРТУ");
        } else if (err.message.includes("404")) {
            showError("error-msg", "ПОЗЫВНОЙ НЕ НАЙДЕН");
        } else {
            showError("error-msg", err.message);
        }
    }
}

// Enter key on code input
document.addEventListener("DOMContentLoaded", () => {
    const codeInput = document.getElementById("session-code");
    if (codeInput) {
        codeInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") joinSession();
        });
    }

    // Session page init
    const chatArea = document.getElementById("chat-area");
    if (chatArea) {
        initSessionPage();
    }

    // Question input enter key
    const qInput = document.getElementById("question-input");
    if (qInput) {
        qInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") askQuestion();
        });
    }

    // Admin PIN enter key
    const pinInput = document.getElementById("pin-input");
    if (pinInput) {
        pinInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") adminLogin();
        });
    }
});

// --- Session Page ---
function initSessionPage() {
    const raw = sessionStorage.getItem("session");
    if (!raw) {
        window.location.href = "/";
        return;
    }
    currentSession = JSON.parse(raw);

    // Set badge
    document.getElementById("badge-code").textContent = currentSession.code;

    // Set mission info
    document.getElementById("mission-topic").textContent = currentSession.topic;

    // Set role
    const roleNames = { optimist: "ТЕХНООПТИМИСТ", pessimist: "ТЕХНОПЕССИМИСТ" };
    document.getElementById("role-name").textContent = roleNames[currentSession.role];

    // Apply role class for theming
    document.body.classList.add(`role-${currentSession.role}`);

    // Start heartbeat
    heartbeatInterval = setInterval(sendHeartbeat, 30000);
    sendHeartbeat();
}

async function sendHeartbeat() {
    if (!currentSession) return;
    try {
        await api("POST", "/api/sessions/heartbeat", { code: currentSession.code });
    } catch (err) {
        // Silently ignore heartbeat errors
    }
}

const SECTION_TITLES = [
    "СУТЬ ПОЗИЦИИ",
    "КЛЮЧЕВЫЕ АРГУМЕНТЫ",
    "ЧТО ПРОИЗОЙДЁТ К 2100",
    "ГЛАВНЫЕ РИСКИ",
    "ЧТО СКАЗАТЬ НА ДЕБАТАХ",
    "ВОПРОСЫ ОППОНЕНТАМ",
    "НОВОСТЬ ИЗ 2100 ГОДА",
];

const SECTION_KEYS = [
    "position", "arguments", "predictions", "risks",
    "debate_speech", "opponent_questions", "news_2100",
];

function renderAnswer(answer) {
    let html = "";
    for (let i = 0; i < SECTION_KEYS.length; i++) {
        const text = answer[SECTION_KEYS[i]];
        if (!text) continue;
        html += `<div class="section-block">
            <div class="section-title">${SECTION_TITLES[i]}</div>
            <div class="section-content">${escapeHtml(text)}</div>
        </div>`;
    }
    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

async function askQuestion() {
    if (!currentSession) return;
    const input = document.getElementById("question-input");
    const question = input.value.trim();
    if (!question) return;

    // Remove welcome message
    const welcome = document.getElementById("welcome-msg");
    if (welcome) welcome.remove();

    // Disable input
    input.value = "";
    input.disabled = true;
    document.getElementById("send-btn").disabled = true;
    document.getElementById("loading-indicator").style.display = "block";

    // Add question to chat
    const chatArea = document.getElementById("chat-area");

    try {
        const data = await api("POST", "/api/ask", {
            code: currentSession.code,
            question: question,
        });

        const block = document.createElement("div");
        block.className = "qa-block";
        block.innerHTML = `
            <div class="qa-question">
                <span class="prompt-symbol">&gt;</span> ${escapeHtml(question)}
            </div>
            <div class="qa-answer">
                ${renderAnswer(data.answer)}
            </div>
        `;
        chatArea.appendChild(block);
        block.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        const errBlock = document.createElement("div");
        errBlock.className = "qa-block";
        errBlock.innerHTML = `
            <div class="qa-question">
                <span class="prompt-symbol">&gt;</span> ${escapeHtml(question)}
            </div>
            <div class="qa-answer" style="color: var(--accent);">
                ОШИБКА СВЯЗИ: ${escapeHtml(err.message)}
            </div>
        `;
        chatArea.appendChild(errBlock);
    } finally {
        input.disabled = false;
        document.getElementById("send-btn").disabled = false;
        document.getElementById("loading-indicator").style.display = "none";
        input.focus();
    }
}

// --- Admin Page ---
async function adminLogin() {
    const pin = document.getElementById("pin-input").value;
    try {
        await api("POST", "/api/admin/login", { pin });
        document.getElementById("pin-overlay").style.display = "none";
        document.getElementById("dashboard").style.display = "block";
        refreshDashboard();
    } catch (err) {
        showError("pin-error", "НЕВЕРНЫЙ КОД ДОСТУПА");
    }
}

async function createAllSessions() {
    try {
        const data = await api("POST", "/api/sessions/create-all");
        refreshDashboard();
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

async function refreshDashboard() {
    try {
        const sessions = await api("GET", "/api/sessions");
        renderSessionsGrid(sessions);
    } catch (err) {
        // No sessions yet
    }
}

function renderSessionsGrid(sessions) {
    const grid = document.getElementById("sessions-grid");
    if (!grid) return;

    grid.innerHTML = sessions.map(s => {
        const roleClass = s.role === "optimist" ? "card-optimist" : "card-pessimist";
        const roleLabel = s.role === "optimist" ? "ОПТИМИСТ" : "ПЕССИМИСТ";
        const statusLabel = { waiting: "Ожидает", active: "Онлайн", disconnected: "Отключена" };
        return `<div class="session-card ${roleClass}" onclick="openSessionDetail('${s.code}')">
            <div class="card-code">${s.code}</div>
            <div class="card-topic">${escapeHtml(s.topic)}</div>
            <div class="card-status">
                <span class="status-dot ${s.status}"></span>
                <span>${statusLabel[s.status] || s.status}</span>
                <span style="margin-left:auto">${roleLabel}</span>
            </div>
            <div class="card-questions">Вопросов: ${s.question_count}</div>
        </div>`;
    }).join("");
}

async function openSessionDetail(code) {
    try {
        const data = await api("GET", `/api/sessions/${code}/history`);
        document.getElementById("detail-title").textContent =
            `${data.code} — ${data.topic}`;

        const historyEl = document.getElementById("detail-history");
        if (data.questions.length === 0) {
            historyEl.innerHTML = '<p style="color:var(--text-dim)">Вопросов пока нет.</p>';
        } else {
            historyEl.innerHTML = data.questions.map(q => `
                <div class="detail-qa">
                    <div class="detail-question">&gt; ${escapeHtml(q.question)}</div>
                    <div class="detail-answer">${escapeHtml(q.answer.position || '')}</div>
                </div>
            `).join("");
        }

        document.getElementById("session-detail").style.display = "block";
    } catch (err) {
        alert("Ошибка: " + err.message);
    }
}

function closeDetail() {
    document.getElementById("session-detail").style.display = "none";
}
```

- [ ] **Step 2: Commit**

```bash
git add static/app.js
git commit -m "feat: client-side logic for entry, session, and admin pages"
```

---

## Task 13: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

File `tests/test_integration.py`:
```python
"""
End-to-end integration test.
Tests the full flow: create sessions -> join -> ask question -> verify history.
Mocks OpenRouter LLM calls but uses real indexing on test fixtures.
"""
import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport


MOCK_LLM_RESPONSE = """## 1. Суть позиции
Человечество к 2100 году станет многопланетным видом.

## 2. Ключевые аргументы
1. В произведении "Туманность Андромеды" описаны орбитальные станции.
2. Стоимость запусков снижается.
3. Роботы могут осваивать среды.
4. Лунные базы как промежуточный этап.
5. Многопланетность как страховка.

## 3. Что произойдёт к 2100
- Постоянная база на Луне
- Орбитальные заводы
- Миссии к Марсу
- Добыча ресурсов

## 4. Главные риски
- Высокая стоимость
- Радиация
- Конфликты
- Мусор

## 5. Что сказать на дебатах
Космос — это следующий шаг. Как океан стал пространством торговли, космос станет рабочей средой.

## 6. Вопросы оппонентам
1. Как снижать риски без космоса?
2. Когда барьеры были непреодолимы?

## 7. Новость из 2100 года
Выпускники лунной школы запустили станцию переработки реголита."""


@pytest.fixture
def integration_app(optimist_dir, pessimist_dir):
    """Build app with real indexed test fixtures."""
    import config
    # Point collections at test fixtures
    original_dir = config.COLLECTIONS_DIR

    import tempfile, shutil, os
    tmpdir = tempfile.mkdtemp()
    shutil.copytree(optimist_dir, os.path.join(tmpdir, "optimist"))
    shutil.copytree(pessimist_dir, os.path.join(tmpdir, "pessimist"))
    config.COLLECTIONS_DIR = tmpdir

    # Re-import serve to trigger fresh state
    import importlib
    import serve
    importlib.reload(serve)

    yield serve.app

    config.COLLECTIONS_DIR = original_dir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_flow(integration_app):
    transport = ASGITransport(app=integration_app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create sessions
        resp = await client.post("/api/sessions/create-all")
        assert resp.status_code == 200
        codes = resp.json()["codes"]
        assert "ЛУНА-01" in codes

        # 2. Join as optimist team
        resp = await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "optimist"

        # 3. Ask a question (mock LLM)
        with patch("agents.call_openrouter", new_callable=AsyncMock, return_value=MOCK_LLM_RESPONSE):
            resp = await client.post("/api/ask", json={
                "code": "ЛУНА-01",
                "question": "Станет ли человечество космической цивилизацией?"
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "многопланетным" in data["answer"]["position"]
            assert "Туманность Андромеды" in data["answer"]["arguments"]

        # 4. Check history
        resp = await client.get("/api/sessions/ЛУНА-01/history")
        assert resp.status_code == 200
        history = resp.json()
        assert len(history["questions"]) == 1
        assert history["questions"][0]["question"] == "Станет ли человечество космической цивилизацией?"

        # 5. Check admin view
        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        luna01 = [s for s in sessions if s["code"] == "ЛУНА-01"][0]
        assert luna01["status"] == "active"
        assert luna01["question_count"] == 1

        # 6. Verify second team can't join same session
        resp = await client.post("/api/sessions/join", json={"code": "ЛУНА-01"})
        assert resp.status_code == 409

        # 7. But can join a different session
        resp = await client.post("/api/sessions/join", json={"code": "ЗВЕЗДА-02"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "pessimist"
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest tests/test_integration.py -v
```

Expected: all assertions PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full session flow"
```

---

## Task 14: Final Wiring and Launch Verification

**Files:**
- Verify: all files present and connected

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass across all test files

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Verify server starts without text collections**

```bash
timeout 10 python serve.py || true
```

Expected: server starts, warns about missing .txt files, listens on port 8000

- [ ] **Step 4: Verify server starts with test fixture collections**

Copy test fixtures to data/collections temporarily and verify full startup:

```bash
cp tests/fixtures/optimist/*.txt data/collections/optimist/
cp tests/fixtures/pessimist/*.txt data/collections/pessimist/
timeout 60 python serve.py || true
```

Expected: server indexes collections, prints chunk counts, starts successfully

- [ ] **Step 5: Clean up test data from collections**

```bash
rm data/collections/optimist/*.txt data/collections/pessimist/*.txt
```

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: final wiring and launch verification"
```
