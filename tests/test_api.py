import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from models import AgentResponse


@pytest.fixture
def mock_collections():
    mock_col = MagicMock()
    mock_col.chunks = []
    collections = {"optimist": mock_col, "pessimist": mock_col}
    return collections


@pytest.fixture
def app(mock_collections):
    with patch.dict("serve.collections", mock_collections):
        from serve import app as _app
        yield _app


@pytest.mark.asyncio
async def test_admin_login(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/admin/login", json={"pin": "0000"})
        assert resp.status_code == 401
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
