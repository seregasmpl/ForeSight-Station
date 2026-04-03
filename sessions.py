from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import config
from models import SessionData, AgentResponse, QuestionRecord

logger = logging.getLogger("forsight")

_SECTION_KEYS = ["position", "arguments", "predictions", "risks", "debate_speech", "opponent_questions", "news_2100"]
_SECTION_TITLES = ["Суть позиции", "Ключевые аргументы", "Что произойдёт к 2100", "Главные риски", "Что сказать на дебатах", "Вопросы оппонентам", "Новость из 2100 года"]

_SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "sessions.json")


def _answer_to_text(answer: AgentResponse) -> str:
    parts = []
    for i, (key, title) in enumerate(zip(_SECTION_KEYS, _SECTION_TITLES), 1):
        val = getattr(answer, key, "")
        if val:
            parts.append(f"## {i}. {title}\n{val}")
    return "\n\n".join(parts)


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(_SESSIONS_FILE), exist_ok=True)
        data = []
        for s in self._sessions.values():
            d = s.model_dump(mode="json")
            d["used_chunk_ids"] = list(s.used_chunk_ids)
            data.append(d)
        try:
            tmp = _SESSIONS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _SESSIONS_FILE)
        except Exception as e:
            logger.warning("Failed to save sessions: %s", e)

    def _load(self):
        if not os.path.exists(_SESSIONS_FILE):
            return
        try:
            with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                d["used_chunk_ids"] = set(d.get("used_chunk_ids", []))
                session = SessionData.model_validate(d)
                # Mark previously-active sessions as disconnected on cold start
                if session.status == "active":
                    session.status = "disconnected"
                self._sessions[session.code] = session
            logger.info("Loaded %d sessions from disk", len(self._sessions))
        except Exception as e:
            logger.warning("Failed to load sessions: %s", e)

    # ── session management ────────────────────────────────────────────────────

    def _make_code(self, topic: str, role: str) -> str:
        prefix = config.TOPIC_PREFIXES[topic]
        suffix = config.ROLE_SUFFIXES[role]
        return f"{prefix}-{suffix}"

    def create_session(self, topic: str, role: str, force: bool = False) -> str:
        code = self._make_code(topic, role)
        existing = self._sessions.get(code)
        if existing and not force:
            return code
        self._sessions[code] = SessionData(code=code, topic=topic, role=role)
        self._save()
        return code

    def create_all_sessions(self, force: bool = False) -> list[str]:
        codes = []
        for topic in config.TOPICS:
            for role in ("optimist", "pessimist"):
                codes.append(self.create_session(topic, role, force=force))
        self._save()
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
            return None
        session.status = "active"
        session.connected_at = datetime.now()
        session.last_heartbeat = datetime.now()
        self._save()
        return session

    def heartbeat(self, code: str) -> bool:
        session = self._sessions.get(code)
        if session is None or session.status != "active":
            return False
        session.last_heartbeat = datetime.now()
        return True  # no save — too frequent, not critical

    def check_stale_sessions(self, timeout_seconds: int = 120):
        changed = False
        now = datetime.now()
        for session in self._sessions.values():
            if session.status != "active":
                continue
            if session.last_heartbeat is None:
                continue
            if (now - session.last_heartbeat).total_seconds() > timeout_seconds:
                session.status = "disconnected"
                changed = True
        if changed:
            self._save()

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
        session.chat_messages.append({"role": "user", "content": question})
        session.chat_messages.append({"role": "assistant", "content": _answer_to_text(answer)})
        self._save()

    def get_history_summary(self, code: str) -> str:
        session = self._sessions.get(code)
        if session is None or not session.questions:
            return ""
        summaries = []
        for q in session.questions:
            summaries.append(f"Вопрос: {q.question}\nТезис ответа: {q.answer.position}")
        return "\n\n".join(summaries)

    def get_chat_messages(self, code: str) -> list[dict]:
        session = self._sessions.get(code)
        if session is None:
            return []
        return list(session.chat_messages)
