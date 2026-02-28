"""Parse engine: orchestrate parallel file parsing across language adapters."""

from __future__ import annotations

import importlib
from pathlib import Path

from tree_sitter import Language, Parser

from archex.exceptions import ParseError

# Maps language_id → (module_name, function_name)
_LANGUAGE_LOADERS: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
}


class TreeSitterEngine:
    """Manages tree-sitter Language and Parser instances with per-language caching."""

    def __init__(self) -> None:
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}

    def get_language(self, language_id: str) -> Language:
        """Return a cached Language for the given language_id.

        Raises ParseError if the language is not supported or the module is missing.
        """
        if language_id in self._languages:
            return self._languages[language_id]

        if language_id not in _LANGUAGE_LOADERS:
            raise ParseError(f"Unsupported language: {language_id!r}")

        module_name, func_name = _LANGUAGE_LOADERS[language_id]
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ParseError(
                f"tree-sitter grammar for {language_id!r} not installed: {exc}"
            ) from exc

        try:
            func = getattr(module, func_name)
            lang = Language(func())  # pyright: ignore[reportDeprecated]
        except Exception as exc:
            raise ParseError(f"Failed to load tree-sitter language {language_id!r}: {exc}") from exc

        self._languages[language_id] = lang
        return lang

    def get_parser(self, language_id: str) -> Parser:
        """Return a cached Parser for the given language_id."""
        if language_id in self._parsers:
            return self._parsers[language_id]

        lang = self.get_language(language_id)
        parser = Parser(lang)
        self._parsers[language_id] = parser
        return parser

    def parse_file(
        self, file_path: str | Path, language_id: str, max_file_size: int = 10_000_000
    ) -> object:
        """Read a file and return its parse tree.

        Raises ParseError on IO errors, unsupported language, or files exceeding max_file_size.
        """
        path = Path(file_path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ParseError(f"Failed to stat file {file_path!r}: {exc}") from exc
        if size > max_file_size:
            raise ParseError(f"File {file_path!r} exceeds maximum size limit")
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise ParseError(f"Failed to read file {file_path!r}: {exc}") from exc
        return self.parse_bytes(source, language_id)

    def parse_bytes(self, source: bytes, language_id: str) -> object:
        """Parse raw bytes and return the tree-sitter Tree object.

        Return type is object for pyright strict compatibility (no tree-sitter stubs).
        """
        parser = self.get_parser(language_id)
        return parser.parse(source)
