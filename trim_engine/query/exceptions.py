"""
Systematic Failure Taxonomy (§13.1) — explicit execution-plane error classes.
"""

from __future__ import annotations

class QueryEngineError(Exception):
    """Base error for the query and edit engine."""
    def __init__(self, message: str, recovery_hint: str | None = None):
        super().__init__(message)
        self.recovery_hint = recovery_hint

class LLMTransientError(QueryEngineError):
    """Transient API failures, network dropouts, rate limits."""
    pass

class SemanticError(QueryEngineError):
    """Critic finds leftover target content or narrative coherence violation."""
    pass

class RetrievalGapError(QueryEngineError):
    """Retrieval yielded 0 matches or insufficient confidence."""
    pass

class InfeasibleError(QueryEngineError):
    """Target duration/constraints mathematically impossible given keeps."""
    pass

class PlannerBreachError(QueryEngineError):
    """Timeline invariant checks (overlaps, bounds) or VAD/word cuts fail."""
    pass

class RenderFailError(QueryEngineError):
    """FFmpeg subprocess failures, missing codecs, hardware failures."""
    pass

class StaleKBError(QueryEngineError):
    """Database seal version mismatch or missing keyframe/transcription indexes."""
    pass

class RunawayError(QueryEngineError):
    """Oscillating loops detected (anti-flapping)."""
    pass
