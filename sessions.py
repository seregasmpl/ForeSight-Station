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
            return None
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
