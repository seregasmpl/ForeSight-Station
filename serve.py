from __future__ import annotations

import asyncio
import logging
import os
import html

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

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"игнорируй.{0,30}инструкц",
    r"забудь.{0,20}(кто ты|роль|инструкц)",
    r"притворись.{0,30}(другой|не|без)",
    r"ты теперь",
    r"новая роль",
    r"действуй без ограничений",
    r"без цензуры",
    r"покажи.{0,20}(промт|инструкц|систем)",
    r"ignore.{0,30}(all.{0,10})?previous.{0,10}instructions?",
    r"ignore.{0,30}instructions?",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as.{0,20}(different|another|without)",
    r"jailbreak",
    r"dan mode",
    r"developer mode",
    r"system prompt",
    r"disregard",
]

import re as _re
_INJECTION_RE = _re.compile(
    "|".join(_INJECTION_PATTERNS),
    _re.IGNORECASE,
)


def _check_injection(text: str) -> bool:
    """Returns True if the text looks like a prompt injection attempt."""
    return bool(_INJECTION_RE.search(text))

# --- Global state ---
session_manager = SessionManager()
collections: dict[str, IndexedCollection] = {}
retrievers: dict[str, HybridRetriever] = {}

# Locks for concurrent access safety
_retriever_lock = asyncio.Lock()   # embed model is not thread-safe
_session_lock = asyncio.Lock()     # session mutations + disk writes


async def process_question(code: str, question: str) -> AgentResponse:
    session = session_manager.get_session(code)
    if session is None:
        raise HTTPException(404, "НЕ НАЙДЕН")

    retriever = retrievers.get(session.role)
    if retriever is None:
        raise HTTPException(500, f"Collection '{session.role}' not indexed")

    # Serialize retriever access (SentenceTransformer.encode not thread-safe)
    # and run off event loop so heartbeats / static files keep working
    async with _retriever_lock:
        chunks = await asyncio.to_thread(
            retriever.search,
            query=question,
            k=config.MMR_TOP_K,
            used_chunk_ids=session.used_chunk_ids,
        )
    chunk_ids = [c.id for c in chunks]

    history_summary = session_manager.get_history_summary(code)
    prior_messages = session_manager.get_chat_messages(code)
    agent = Agent(role=session.role)
    answer = await agent.ask(
        question=question,
        topic=session.topic,
        chunks=chunks,
        history_summary=history_summary,
        prior_messages=prior_messages,
    )

    # Serialize session state mutations + atomic disk write
    async with _session_lock:
        session_manager.add_question(code, question, answer, chunk_ids)
    return answer


async def heartbeat_checker():
    while True:
        session_manager.check_stale_sessions(timeout_seconds=config.HEARTBEAT_TIMEOUT)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    task = asyncio.create_task(heartbeat_checker())
    logger.info("Server ready!")
    yield
    task.cancel()


app = FastAPI(title="Форсайт-Станция", lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index_page():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/session")
async def session_page():
    return FileResponse(os.path.join(static_dir, "session.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(static_dir, "admin.html"))


@app.get("/viewer")
async def viewer_page():
    return FileResponse(os.path.join(static_dir, "viewer.html"))


@app.post("/api/admin/login")
async def admin_login(body: AdminLogin):
    if body.pin != config.ADMIN_PIN:
        raise HTTPException(401, "Wrong PIN")
    return {"ok": True}


@app.post("/api/sessions/create-all")
async def create_all_sessions():
    codes = session_manager.create_all_sessions()
    return {"codes": codes}


@app.post("/api/sessions/reset-all")
async def reset_all_sessions():
    codes = session_manager.create_all_sessions(force=True)
    return {"codes": codes}


@app.post("/api/sessions/join")
async def join_session(body: SessionJoin):
    session = session_manager.join_session(body.code)
    if session is None:
        existing = session_manager.get_session(body.code)
        if existing is None:
            raise HTTPException(404, "НЕ НАЙДЕН")
        raise HTTPException(409, "УЖЕ НА БОРТУ")
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
        raise HTTPException(404, "НЕ НАЙДЕН")
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


@app.post("/api/ask")
async def ask(body: AskRequest):
    session = session_manager.get_session(body.code)
    if session is None:
        raise HTTPException(404, "НЕ НАЙДЕН")
    if session.status != "active":
        raise HTTPException(403, "Session not active")
    if not body.question.strip():
        raise HTTPException(400, "Вопрос не может быть пустым")
    if _check_injection(body.question):
        logger.warning("Injection attempt blocked from session %s: %.80s", body.code, body.question)
        raise HTTPException(400, "Вопрос не по теме форсайт-сессии")

    answer = await process_question(body.code, body.question)
    return {
        "question": html.escape(body.question),
        "answer": answer.model_dump(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host=config.HOST, port=config.PORT, reload=True)
