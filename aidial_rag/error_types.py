from enum import StrEnum, auto


class ErrorType(StrEnum):
    """Machine-readable error types for the indexing and retrieval processes."""

    INDEX_MISSING = auto()
    INDEX_INCOMPATIBLE = auto()
