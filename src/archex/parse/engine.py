"""Parse engine: orchestrate parallel file parsing across language adapters."""

from __future__ import annotations

import importlib
from pathlib import Path

from tree_sitter import Language, Parser

from archex.exceptions import ParseError
from archex.languages import LANGUAGE_PACK_NAMES

# Maps language_id → (preferred module_name, function_name).
_PREFERRED_LANGUAGE_LOADERS: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "swift": ("tree_sitter_swift", "language"),
}


def _coerce_tree_sitter_language(language_id: str, value: object, source: str) -> Language:
    if isinstance(value, Language):
        return value

    try:
        return Language(value)  # pyright: ignore[reportDeprecated]
    except TypeError as exc:
        value_type = type(value)
        install_hint = (
            "; install tree-sitter-language-pack<1.0"
            if source == "tree-sitter-language-pack"
            else ""
        )
        raise ParseError(
            f"incompatible {source} grammar for {language_id!r}: expected a "
            f"tree_sitter.Language-compatible value, got "
            f"{value_type.__module__}.{value_type.__qualname__}{install_hint}"
        ) from exc


class TreeSitterEngine:
    """Manages tree-sitter Language and Parser instances with per-language caching."""

    def __init__(self) -> None:
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}

    def get_language(self, language_id: str) -> Language:
        """Return a cached Language for the given language_id.

        Raises ParseError if the language is not supported or the grammar cannot be loaded.
        """
        if language_id in self._languages:
            return self._languages[language_id]

        if language_id not in LANGUAGE_PACK_NAMES:
            raise ParseError(f"Unsupported language: {language_id!r}")

        if language_id in _PREFERRED_LANGUAGE_LOADERS:
            lang = self._load_preferred_language(language_id)
        else:
            lang = self._try_language_pack(language_id)
        self._languages[language_id] = lang
        return lang

    def _load_preferred_language(self, language_id: str) -> Language:
        module_name, func_name = _PREFERRED_LANGUAGE_LOADERS[language_id]
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return self._try_language_pack(language_id)

        try:
            func = getattr(module, func_name)
            return _coerce_tree_sitter_language(language_id, func(), module_name)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Failed to load tree-sitter language {language_id!r}: {exc}") from exc

    def _try_language_pack(self, language_id: str) -> Language:
        """Load grammar from tree-sitter-language-pack after standalone lookup misses."""
        pack_name = LANGUAGE_PACK_NAMES[language_id]
        try:
            from tree_sitter_language_pack import (
                get_language as _pack_get_language,  # type: ignore[import-untyped]
            )

            lang = _pack_get_language(pack_name)  # pyright: ignore[reportUnknownVariableType,reportArgumentType]
            return _coerce_tree_sitter_language(language_id, lang, "tree-sitter-language-pack")
        except ImportError as exc:
            raise ParseError(
                f"tree-sitter grammar for {language_id!r} not installed "
                f"(tried standalone package and tree-sitter-language-pack): {exc}"
            ) from exc
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(
                f"Failed to load tree-sitter language {language_id!r} from language-pack: {exc}"
            ) from exc

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
