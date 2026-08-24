from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryType(str, Enum):
    semantic = "semantic"
    episodic = "episodic"
    procedural = "procedural"


class Memory(BaseModel):
    id: str
    type: MemoryType
    content: str
    keywords: list[str] = Field(default_factory=list)
    importance: float
    confidence: float
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_used_at: datetime | None = None
    use_count: int = 0
    expires_at: datetime | None = None


class MemoryScope(BaseModel):
    version: int = 1
    scope: str
    memories: list[Memory] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """Structured rolling summary for one channel session."""

    topic: str | None = None
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.topic or self.facts or self.decisions or self.unresolved)


class MemoryCandidate(BaseModel):
    type: MemoryType
    content: str
    importance: float = 0.5
    confidence: float = 0.5


class ExtractionResult(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list)


class RetrievedMemory(BaseModel):
    memory: Memory
    score: float
    reason: str = ""
