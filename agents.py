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
        source_label = chunk.author if chunk.author else chunk.source_file
        chunks_text += f"[Фрагмент {i} — {source_label}]\n{chunk.text}\n\n"

    if not history_summary:
        history_summary = "Это первый вопрос команды."

    return template.format(
        topic=topic,
        chunks=chunks_text.strip(),
        focus_lens=focus_lens,
        history=history_summary,
    )


_SECTION_HEADER_RE = re.compile(
    r"^(суть позиции|ключевые аргументы|что произойдёт к 2100|главные риски|"
    r"что сказать на дебатах|вопросы оппонентам|новость из 2100 года)\s*\n",
    re.IGNORECASE,
)


def _strip_echo_header(text: str) -> str:
    """LLM sometimes repeats section titles as the first lines of content; strip all of them."""
    prev = None
    while prev != text:
        prev = text
        text = _SECTION_HEADER_RE.sub("", text, count=1).strip()
    return text


def parse_response(raw: str) -> AgentResponse:
    parts = re.split(r"##\s*\d+\.\s*", raw)
    # parts[0] is always pre-header content (empty or LLM preamble) — skip it
    sections = [_strip_echo_header(s) for s in parts[1:] if s.strip()]

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
    return AgentResponse(
        position=raw,
        arguments="",
        predictions="",
        risks="",
        debate_speech="",
        opponent_questions="",
        news_2100="",
    )


async def call_openrouter(
    system_prompt: str,
    user_message: str,
    prior_messages: list[dict] | None = None,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if prior_messages:
        messages.extend(prior_messages)
    messages.append({"role": "user", "content": user_message})

    import asyncio as _asyncio

    async with httpx.AsyncClient() as client:
        last_err = None
        for attempt in range(3):
            try:
                response = await client.post(
                    config.OPENROUTER_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": config.OPENROUTER_MODEL,
                        "messages": messages,
                        "temperature": config.LLM_TEMPERATURE,
                    },
                    timeout=config.LLM_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_err = e
                if attempt < 2:
                    await _asyncio.sleep(2)
        raise last_err


class Agent:
    def __init__(self, role: str):
        self.role = role

    async def ask(
        self,
        question: str,
        topic: str,
        chunks: list[Chunk],
        history_summary: str,
        prior_messages: list[dict] | None = None,
    ) -> AgentResponse:
        focus_lens = random.choice(config.FOCUS_LENSES)
        system_prompt = build_system_prompt(
            role=self.role,
            topic=topic,
            chunks=chunks,
            focus_lens=focus_lens,
            history_summary=history_summary,
        )
        raw = await call_openrouter(system_prompt, question, prior_messages=prior_messages)
        return parse_response(raw)
