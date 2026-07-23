from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from archex.exceptions import ParseError
from archex.parse.engine import TreeSitterEngine

PYTHON_SOURCE = b"def hello():\n    pass\n"


def _root(tree: object) -> Any:
    t: Any = tree
    return t.root_node


def test_parse_python_source() -> None:
    engine = TreeSitterEngine()
    tree = engine.parse_bytes(PYTHON_SOURCE, "python")
    root = _root(tree)
    assert root is not None
    assert root.type == "module"


def test_parse_bytes_returns_tree_with_children() -> None:
    engine = TreeSitterEngine()
    source = b"class Foo:\n    def bar(self):\n        pass\n"
    tree = engine.parse_bytes(source, "python")
    root = _root(tree)
    assert len(root.children) > 0


def test_parser_caching_returns_same_instance() -> None:
    engine = TreeSitterEngine()
    parser1 = engine.get_parser("python")
    parser2 = engine.get_parser("python")
    assert parser1 is parser2


def test_language_caching_returns_same_instance() -> None:
    engine = TreeSitterEngine()
    lang1 = engine.get_language("python")
    lang2 = engine.get_language("python")
    assert lang1 is lang2


def test_unsupported_language_raises_parse_error() -> None:
    engine = TreeSitterEngine()
    with pytest.raises(ParseError, match="Unsupported language"):
        engine.get_language("cobol")


def test_parse_file(tmp_path: Path) -> None:
    source_file = tmp_path / "sample.py"
    source_file.write_bytes(PYTHON_SOURCE)
    engine = TreeSitterEngine()
    tree = engine.parse_file(str(source_file), "python")
    root = _root(tree)
    assert root.type == "module"


def test_parse_file_missing_raises_parse_error(tmp_path: Path) -> None:
    engine = TreeSitterEngine()
    with pytest.raises(ParseError, match="Failed to"):
        engine.parse_file(str(tmp_path / "nonexistent.py"), "python")


def test_parse_file_exceeds_max_size_raises_parse_error(tmp_path: Path) -> None:
    """A file larger than max_file_size raises ParseError."""
    source_file = tmp_path / "big.py"
    source_file.write_bytes(b"x = 1\n" * 100)
    engine = TreeSitterEngine()
    with pytest.raises(ParseError, match="exceeds maximum size limit"):
        engine.parse_file(str(source_file), "python", max_file_size=10)


def test_parse_file_at_limit_succeeds(tmp_path: Path) -> None:
    """A file exactly at max_file_size is parsed without error."""
    content = b"x = 1\n"
    source_file = tmp_path / "small.py"
    source_file.write_bytes(content)
    engine = TreeSitterEngine()
    # size == max_file_size is allowed (only strictly greater raises)
    tree = engine.parse_file(str(source_file), "python", max_file_size=len(content))
    root: Any = tree  # type: ignore[assignment]
    assert root.root_node is not None


def test_read_and_parse_file_returns_tree_and_source(tmp_path: Path) -> None:
    source_file = tmp_path / "sample.py"
    source_file.write_bytes(PYTHON_SOURCE)
    engine = TreeSitterEngine()
    tree, source = engine.read_and_parse_file(str(source_file), "python")
    assert source == PYTHON_SOURCE
    assert _root(tree).type == "module"


def test_read_and_parse_file_missing_raises_parse_error(tmp_path: Path) -> None:
    engine = TreeSitterEngine()
    with pytest.raises(ParseError, match="Failed to"):
        engine.read_and_parse_file(str(tmp_path / "nonexistent.py"), "python")


def test_read_and_parse_file_exceeds_max_size_raises_parse_error(tmp_path: Path) -> None:
    source_file = tmp_path / "big.py"
    source_file.write_bytes(b"x = 1\n" * 100)
    engine = TreeSitterEngine()
    with pytest.raises(ParseError, match="exceeds maximum size limit"):
        engine.read_and_parse_file(str(source_file), "python", max_file_size=10)


def test_read_and_parse_file_read_error_raises_parse_error(tmp_path: Path) -> None:
    from unittest.mock import patch

    source_file = tmp_path / "unreadable.py"
    source_file.write_bytes(b"x = 1")
    engine = TreeSitterEngine()
    with (
        patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied")),
        pytest.raises(ParseError, match="Failed to read"),
    ):
        engine.read_and_parse_file(str(source_file), "python")


def test_parse_file_reads_source_exactly_once(tmp_path: Path) -> None:
    """parse_file() (and by extension read_and_parse_file()) must read a file's
    bytes exactly once — the double-read this method exists to eliminate at
    every call site would otherwise silently creep back in."""
    from unittest.mock import patch

    source_file = tmp_path / "sample.py"
    source_file.write_bytes(PYTHON_SOURCE)
    engine = TreeSitterEngine()
    with patch("pathlib.Path.read_bytes", wraps=Path.read_bytes, autospec=True) as mock_read_bytes:
        engine.read_and_parse_file(str(source_file), "python")
    mock_read_bytes.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases: ImportError for grammar module, Language constructor failure,
# and OSError reading file bytes
# ---------------------------------------------------------------------------


def test_missing_grammar_module_raises_parse_error() -> None:
    from unittest.mock import patch

    engine = TreeSitterEngine()
    engine._languages.clear()  # pyright: ignore[reportPrivateUsage]
    engine._parsers.clear()  # pyright: ignore[reportPrivateUsage]
    with (
        patch.object(
            TreeSitterEngine,
            "_try_language_pack",
            side_effect=ParseError("not installed"),
        ),
        patch("importlib.import_module", side_effect=ImportError("no module")),
        pytest.raises(ParseError, match="not installed"),
    ):
        engine.get_language("python")


def test_language_constructor_failure_raises_parse_error() -> None:
    from unittest.mock import MagicMock, patch

    engine = TreeSitterEngine()
    engine._languages.clear()  # pyright: ignore[reportPrivateUsage]
    engine._parsers.clear()  # pyright: ignore[reportPrivateUsage]
    mock_module = MagicMock()
    mock_module.language.side_effect = RuntimeError("bad language")
    with (
        patch("importlib.import_module", return_value=mock_module),
        pytest.raises(ParseError, match="Failed to load"),
    ):
        engine.get_language("python")


def test_language_pack_incompatible_language_raises_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    class Language:
        pass

    def get_language(_name: str) -> object:
        return Language()

    mock_pack = SimpleNamespace(get_language=get_language)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", mock_pack)

    engine = TreeSitterEngine()
    with pytest.raises(ParseError, match="incompatible tree-sitter-language-pack"):
        engine.get_parser("c")


def test_parse_file_read_error_raises_parse_error(tmp_path: Path) -> None:
    from unittest.mock import patch

    source_file = tmp_path / "unreadable.py"
    source_file.write_bytes(b"x = 1")
    engine = TreeSitterEngine()
    with (
        patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied")),
        pytest.raises(ParseError, match="Failed to read"),
    ):
        engine.parse_file(str(source_file), "python")
