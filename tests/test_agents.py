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
    assert resp.position == raw


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
