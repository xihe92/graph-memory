"""Graph memory for coding agents: a knowledge graph distilled from agent sessions.

A session's exchanges wait in a small buffer until they graduate into the knowledge graph;
only the knowledge is kept for good, and recall reads only the knowledge.
"""

from .anchors import Access, CodeAnchor, SourceRef
from .buffer import Exchange, SessionBuffer, Summary
from .embedding import LocalEmbedder, default_embedder
from .knowledge import Entity, Item, ItemType, KnowledgeStore, Relation, Source
from .models import CommandModel, claude_code_model
from .vectors import Embedder, JsonVectorIndex

__all__ = [
    "Access",
    "CodeAnchor",
    "CommandModel",
    "Embedder",
    "Entity",
    "Exchange",
    "Item",
    "ItemType",
    "JsonVectorIndex",
    "KnowledgeStore",
    "Relation",
    "SessionBuffer",
    "Source",
    "SourceRef",
    "Summary",
    "default_embedder",
    "LocalEmbedder",
    "claude_code_model",
]
