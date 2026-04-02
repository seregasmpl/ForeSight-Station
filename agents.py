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
        chunks_text += f"[Фрагмент {i} — {chunk.source_file}]\n{chunk.text}\n\n"

    if not history_summary:
        history_summary = "Это первый вопрос команды."

    return template.format(
        topic=topic,
        chunks=chunks_text.strip(),
        focus_lens=focus_lens,
        history=history_summary,
    )


def parse_response(raw: str) -> AgentResponse:
    sections = re.split(r"##\s*\d+\.\s*", raw)
    sections = [s.strip() for s in sections if s.strip()]

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


async def call_openrouter(system_prompt: str, user_message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            config.OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": config.LLM_TEMPERATURE,
            },
            timeout=config.LLM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class Agent:
    def __init__(self, role: str):
        self.role = role

    async def ask(
        self,
        question: str,
        topic: str,
        chunks: list[Chunk],
        history_summary: str,
    ) -> AgentResponse:
        focus_lens = random.choice(config.FOCUS_LENSES)
        system_prompt = build_system_prompt(
            role=self.role,
            topic=topic,
            chunks=chunks,
            focus_lens=focus_lens,
            history_summary=history_summary,
        )
        raw = await call_openrouter(system_prompt, question)
        return parse_response(raw)
