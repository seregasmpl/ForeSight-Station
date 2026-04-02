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


@app.post("/api/admin/login")
async def admin_login(body: AdminLogin):
    if body.pin != config.ADMIN_PIN:
        raise HTTPException(401, "Wrong PIN")
    return {"ok": True}


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
