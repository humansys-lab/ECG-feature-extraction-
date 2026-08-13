"""Governed ECG knowledge library.

This package is never registered in the diagnostic tool registry. It provides
an offline/search CLI, a survey-to-hypothesis knowledge navigator, and an
explicit post-verification challenger. General references never become patient
evidence, and rule/reference-engine conclusions remain excluded at runtime.
"""
from .challenger import (
    CHALLENGE_VERSION,
    KnowledgeChallenger,
    RUNTIME_CATEGORIES,
)
from .index import KnowledgeBase, KnowledgeChunk, SearchResult
from .navigator import KnowledgeNavigator, NAVIGATION_VERSION

__all__ = [
    "CHALLENGE_VERSION",
    "KnowledgeBase",
    "KnowledgeChallenger",
    "KnowledgeChunk",
    "KnowledgeNavigator",
    "NAVIGATION_VERSION",
    "RUNTIME_CATEGORIES",
    "SearchResult",
]
