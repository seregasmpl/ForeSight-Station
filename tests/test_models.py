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
