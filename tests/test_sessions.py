import pytest
from datetime import datetime, timedelta
from sessions import SessionManager
import os


@pytest.fixture
def sm():
    # Clean persisted state to avoid loading active sessions from disk
    sessions_file = os.path.join(os.path.dirname(__file__), "..", "data", "sessions.json")
    if os.path.exists(sessions_file):
        os.remove(sessions_file)
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
    assert result is None


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
