"""What an exchange touched in the codebase, and where it lives in the agent's transcript."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, field_validator


class Access(StrEnum):
    MENTION = "mention"  # named in a prompt
    READ = "read"
    WRITE = "write"


_ACCESS_RANK = {Access.MENTION: 0, Access.READ: 1, Access.WRITE: 2}


class CodeAnchor(BaseModel):
    """Hard link into the codebase: a file, optionally a class/function inside it."""

    model_config = ConfigDict(frozen=True)

    path: str
    symbol: str | None = None
    access: Access = Access.MENTION

    @field_validator("path")
    @classmethod
    def _norm_path(cls, v: str) -> str:
        v = v.strip().replace("\\", "/")
        if not v:
            raise ValueError("path must be non-empty")
        p = PurePosixPath(v).as_posix()
        return p[2:] if p.startswith("./") else p

    @field_validator("symbol")
    @classmethod
    def _norm_symbol(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split())
        return v or None

    @property
    def key(self) -> tuple[str, str]:
        """Identity for matching; access level is deliberately not part of it."""
        return (self.path, self.symbol or "")

    def __str__(self) -> str:
        return f"{self.path}::{self.symbol}" if self.symbol else self.path


class SourceRef(BaseModel):
    """Where the raw turn lives in the agent's own transcript (tool output stays there)."""

    model_config = ConfigDict(frozen=True)

    path: str | None = None
    first: str | None = None  # transcript entry id where the turn starts
    last: str | None = None  # transcript entry id where it ends
