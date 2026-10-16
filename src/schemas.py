"""Shared data types used by every module. Change only with team agreement."""

from typing import Literal, TypedDict

from pydantic import BaseModel

Domain = Literal["hr", "it", "finance"]
Language = Literal["en", "ar", "bilingual"]


class Chunk(BaseModel):
    """A piece of a document plus the metadata needed for citations."""

    chunk_id: str
    text: str
    domain: Domain
    source_filename: str
    language: Language
    score: float = 0.0


class ToolCall(BaseModel):
    """Record of one tool invocation."""

    name: str
    input: dict
    output: str


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    question: str
    session_id: str
    run_id: str
    domain: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    retrieved_chunks: list[Chunk]
    tool_calls: list[ToolCall]
    answer: str
    sources: list[str]
    step_count: int
