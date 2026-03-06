"""Exception hierarchy for archex."""

from __future__ import annotations


class ArchexError(Exception):
    """Base exception for all archex errors."""


class ParseError(ArchexError):
    """Raised when parsing source files fails."""


class AcquireError(ArchexError):
    """Raised when repository acquisition fails."""


class ArchexIndexError(ArchexError):
    """Raised when indexing operations fail."""


class AnalyzeError(ArchexError):
    """Raised when architecture analysis fails."""


class ProviderError(ArchexError):
    """Raised when an LLM provider call fails."""


class CacheError(ArchexError):
    """Raised when cache read/write operations fail."""


class DeltaIndexError(ArchexError):
    """Raised when delta indexing operations fail."""


class ConfigError(ArchexError):
    """Raised when configuration is invalid or missing."""


class LSAPError(ArchexError):
    """Raised when LSAP/LSP client operations fail."""
