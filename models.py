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
