from __future__ import annotations

import tiktoken

from archex.index.chunker import (
    ASTChunker,
    _format_import,  # pyright: ignore[reportPrivateUsage]
    _import_relevant,  # pyright: ignore[reportPrivateUsage]
)
from archex.models import (
    CodeChunk,
    ImportStatement,
    IndexConfig,
    ParsedFile,
    Symbol,
    SymbolKind,
    Visibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SOURCE_SIMPLE = b"""\
import os
from typing import Any

def hello(name: str) -> str:
    \"\"\"Say hello.\"\"\"
    return f"Hello, {name}!"

class Greeter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix} {name}"
"""

# Line numbers (1-indexed):
# 1: import os
# 2: from typing import Any
# 3: (blank)
# 4: def hello(name: str) -> str:
# 5:     """Say hello."""
# 6:     return f"Hello, {name}!"
# 7: (blank)
# 8: class Greeter:
# 9:     def __init__(self, prefix: str) -> None:
# 10:        self.prefix = prefix
# 11: (blank)
# 12:    def greet(self, name: str) -> str:
# 13:        return f"{self.prefix} {name}"

SYMBOLS_SIMPLE = [
    Symbol(
        name="hello",
        qualified_name="hello",
        kind=SymbolKind.FUNCTION,
        file_path="example.py",
        start_line=4,
        end_line=6,
        visibility=Visibility.PUBLIC,
        signature="def hello(name: str) -> str",
    ),
    Symbol(
        name="Greeter",
        qualified_name="Greeter",
        kind=SymbolKind.CLASS,
        file_path="example.py",
        start_line=8,
        end_line=8,
        visibility=Visibility.PUBLIC,
    ),
    Symbol(
        name="Greeter.__init__",
        qualified_name="Greeter.__init__",
        kind=SymbolKind.METHOD,
        file_path="example.py",
        start_line=9,
        end_line=10,
        visibility=Visibility.PUBLIC,
        parent="Greeter",
    ),
    Symbol(
        name="Greeter.greet",
        qualified_name="Greeter.greet",
        kind=SymbolKind.METHOD,
        file_path="example.py",
        start_line=12,
        end_line=13,
        visibility=Visibility.PUBLIC,
        parent="Greeter",
    ),
]

IMPORTS_SIMPLE = [
    ImportStatement(module="os", file_path="example.py", line=1),
    ImportStatement(
        module="typing",
        symbols=["Any"],
        file_path="example.py",
        line=2,
        is_relative=False,
    ),
]

PARSED_SIMPLE = ParsedFile(
    path="example.py",
    language="python",
    symbols=SYMBOLS_SIMPLE,
    imports=IMPORTS_SIMPLE,
    lines=13,
)


def _default_chunker() -> ASTChunker:
    return ASTChunker()


def _find_chunk(chunks: list[CodeChunk], name: str) -> CodeChunk:
    for c in chunks:
        if c.symbol_name == name:
            return c
    raise KeyError(f"No chunk with symbol_name={name!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_function_chunk_contains_full_source() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)

    hello_chunk = _find_chunk(chunks, "hello")
    assert "def hello" in hello_chunk.content
    assert 'return f"Hello, {name}!"' in hello_chunk.content
    assert hello_chunk.start_line == 4
    assert hello_chunk.end_line == 6


def test_import_context_only_relevant_imports() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)

    # "hello" uses neither os nor Any — no import context expected
    hello_chunk = _find_chunk(chunks, "hello")
    # The hello function doesn't reference os or Any directly
    assert "os" not in hello_chunk.imports_context
    assert "Any" not in hello_chunk.imports_context


def test_import_context_included_when_relevant() -> None:
    source = b"import os\n\ndef get_path() -> str:\n    return os.getcwd()\n"
    parsed = ParsedFile(
        path="p.py",
        language="python",
        symbols=[
            Symbol(
                name="get_path",
                qualified_name="get_path",
                kind=SymbolKind.FUNCTION,
                file_path="p.py",
                start_line=3,
                end_line=4,
            )
        ],
        imports=[ImportStatement(module="os", file_path="p.py", line=1)],
        lines=4,
    )
    chunker = _default_chunker()
    chunks = chunker.chunk_file(parsed, source)
    get_path_chunk = _find_chunk(chunks, "get_path")
    assert "import os" in get_path_chunk.imports_context


def test_token_count_is_accurate() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    enc = tiktoken.get_encoding("cl100k_base")

    for chunk in chunks:
        if chunk.imports_context:
            combined = chunk.imports_context + "\n" + chunk.content
        else:
            combined = chunk.content
        expected = len(enc.encode(combined))
        assert chunk.token_count == expected, (
            f"chunk {chunk.id}: expected {expected} tokens, got {chunk.token_count}"
        )


def test_chunk_ids_are_unique() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


def test_chunk_ids_are_deterministic() -> None:
    chunker = _default_chunker()
    chunks_a = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    chunks_b = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]


def test_chunks_sorted_by_start_line() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    lines = [c.start_line for c in chunks]
    assert lines == sorted(lines)


def test_file_level_code_becomes_chunk() -> None:
    # Source with a module-level statement before any symbol
    source = b"MODULE_VAR = 42\n\ndef foo() -> int:\n    return 1\n"
    parsed = ParsedFile(
        path="mod.py",
        language="python",
        symbols=[
            Symbol(
                name="foo",
                qualified_name="foo",
                kind=SymbolKind.FUNCTION,
                file_path="mod.py",
                start_line=3,
                end_line=4,
            )
        ],
        imports=[],
        lines=4,
    )
    chunker = _default_chunker()
    chunks = chunker.chunk_file(parsed, source)

    # There should be a file-level chunk (symbol_name=None) for line 1
    file_chunks = [c for c in chunks if c.symbol_name is None]
    assert file_chunks, "Expected at least one file-level chunk"
    combined_content = " ".join(c.content for c in file_chunks)
    assert "MODULE_VAR" in combined_content


def test_empty_file_produces_no_chunks() -> None:
    source = b""
    parsed = ParsedFile(path="empty.py", language="python", symbols=[], imports=[], lines=0)
    chunker = _default_chunker()
    chunks = chunker.chunk_file(parsed, source)
    assert chunks == []


def test_class_body_and_methods_separate_chunks() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)

    # Greeter class body at line 8 (class declaration)
    greeter_chunk = _find_chunk(chunks, "Greeter")
    assert "class Greeter" in greeter_chunk.content

    # Methods are separate chunks
    init_chunk = _find_chunk(chunks, "Greeter.__init__")
    assert "def __init__" in init_chunk.content

    greet_chunk = _find_chunk(chunks, "Greeter.greet")
    assert "def greet" in greet_chunk.content


def test_large_chunk_gets_split() -> None:
    # Generate a function body large enough to exceed max_tokens=500
    big_body_lines = [f"    x_{i} = {i}  # some assignment" for i in range(200)]
    big_source = "def big_func():\n" + "\n".join(big_body_lines) + "\n"
    source = big_source.encode()

    parsed = ParsedFile(
        path="big.py",
        language="python",
        symbols=[
            Symbol(
                name="big_func",
                qualified_name="big_func",
                kind=SymbolKind.FUNCTION,
                file_path="big.py",
                start_line=1,
                end_line=1 + len(big_body_lines),
            )
        ],
        imports=[],
        lines=1 + len(big_body_lines),
    )
    config = IndexConfig(chunk_max_tokens=50, chunk_min_tokens=5)
    chunker = ASTChunker(config)
    chunks = chunker.chunk_file(parsed, source)

    # Must produce more than one chunk
    sym_chunks = [c for c in chunks if c.symbol_name == "big_func"]
    assert len(sym_chunks) > 1, "Large function should be split into multiple chunks"
    for c in sym_chunks:
        # Allow slight overshoot for single very long lines
        assert c.token_count <= config.chunk_max_tokens + 20


def test_small_adjacent_chunks_merged() -> None:
    # Two tiny file-level lines (comments) that are below min_tokens
    source = b"# line A\n# line B\n# line C\n\ndef foo() -> None:\n    pass\n"
    parsed = ParsedFile(
        path="small.py",
        language="python",
        symbols=[
            Symbol(
                name="foo",
                qualified_name="foo",
                kind=SymbolKind.FUNCTION,
                file_path="small.py",
                start_line=5,
                end_line=6,
            )
        ],
        imports=[],
        lines=6,
    )
    config = IndexConfig(chunk_min_tokens=100, chunk_max_tokens=500)
    chunker = ASTChunker(config)
    chunks = chunker.chunk_file(parsed, source)

    # The tiny comment lines should be merged into a single file-level chunk
    file_chunks = [c for c in chunks if c.symbol_name is None]
    # All file-level content should be in one merged chunk (or at least not three separate chunks)
    assert len(file_chunks) <= 2


def test_chunk_files_across_multiple_files() -> None:
    source_a = b"def a() -> int:\n    return 1\n"
    source_b = b"def b() -> int:\n    return 2\n"

    parsed_a = ParsedFile(
        path="a.py",
        language="python",
        symbols=[
            Symbol(
                name="a",
                qualified_name="a",
                kind=SymbolKind.FUNCTION,
                file_path="a.py",
                start_line=1,
                end_line=2,
            )
        ],
        imports=[],
        lines=2,
    )
    parsed_b = ParsedFile(
        path="b.py",
        language="python",
        symbols=[
            Symbol(
                name="b",
                qualified_name="b",
                kind=SymbolKind.FUNCTION,
                file_path="b.py",
                start_line=1,
                end_line=2,
            )
        ],
        imports=[],
        lines=2,
    )
    chunker = _default_chunker()
    chunks = chunker.chunk_files([parsed_a, parsed_b], {"a.py": source_a, "b.py": source_b})

    file_paths = {c.file_path for c in chunks}
    assert "a.py" in file_paths
    assert "b.py" in file_paths

    # Sorted by (file_path, start_line)
    pairs = [(c.file_path, c.start_line) for c in chunks]
    assert pairs == sorted(pairs)


def test_chunk_file_path_and_language_set() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    for chunk in chunks:
        assert chunk.file_path == "example.py"
        assert chunk.language == "python"


def test_chunk_id_format() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    for chunk in chunks:
        # id format: {file_path}:{symbol_name_or__module}:{start_line}
        parts = chunk.id.split(":")
        assert len(parts) == 3
        assert parts[0] == chunk.file_path
        if chunk.symbol_name is not None:
            assert parts[1] == chunk.symbol_name
        else:
            assert parts[1] == "_module"
        assert parts[2] == str(chunk.start_line)


def test_no_symbols_file_produces_file_level_chunks() -> None:
    source = b"x = 1\ny = 2\nz = x + y\n"
    parsed = ParsedFile(
        path="vars.py",
        language="python",
        symbols=[],
        imports=[],
        lines=3,
    )
    chunker = _default_chunker()
    chunks = chunker.chunk_file(parsed, source)
    assert len(chunks) >= 1
    assert all(c.symbol_name is None for c in chunks)
    combined = " ".join(c.content for c in chunks)
    assert "x = 1" in combined


def test_symbol_kind_set_on_chunk() -> None:
    chunker = _default_chunker()
    chunks = chunker.chunk_file(PARSED_SIMPLE, SOURCE_SIMPLE)
    hello_chunk = _find_chunk(chunks, "hello")
    assert hello_chunk.symbol_kind == SymbolKind.FUNCTION

    greeter_chunk = _find_chunk(chunks, "Greeter")
    assert greeter_chunk.symbol_kind == SymbolKind.CLASS


# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


def test_format_import_with_alias() -> None:
    imp = ImportStatement(module="numpy", alias="np", file_path="x.py", line=1)
    assert _format_import(imp) == "import numpy as np"


def test_format_import_from_with_alias() -> None:
    imp = ImportStatement(
        module="collections",
        symbols=["OrderedDict"],
        alias="OD",
        file_path="x.py",
        line=1,
    )
    assert _format_import(imp) == "from collections import OrderedDict as OD"


def test_import_relevant_bare_import_match() -> None:
    # module="os.path" → base="path"; "path" appears in content
    imp = ImportStatement(module="os.path", file_path="x.py", line=1)
    assert _import_relevant(imp, "result = path.join(a, b)") is True


def test_import_relevant_alias_match() -> None:
    imp = ImportStatement(module="numpy", alias="np", file_path="x.py", line=1)
    assert _import_relevant(imp, "arr = np.array([1, 2, 3])") is True


def test_import_relevant_no_match() -> None:
    imp = ImportStatement(module="sys", file_path="x.py", line=1)
    assert _import_relevant(imp, "print('hello')") is False


def test_merge_backward_into_previous() -> None:
    # Source: header comment, foo function, then a footer comment.
    # With a high min_tokens threshold, both the header and footer are small
    # file-level chunks. The footer (after foo) has no next file-level chunk
    # so _merge_small_chunks falls back to merging it into the previous
    # file-level result entry (the header).
    source = b"# header\n\ndef foo():\n    pass\n\n# footer\n"
    parsed = ParsedFile(
        path="hf.py",
        language="python",
        symbols=[
            Symbol(
                name="foo",
                qualified_name="foo",
                kind=SymbolKind.FUNCTION,
                file_path="hf.py",
                start_line=3,
                end_line=4,
            )
        ],
        imports=[],
        lines=7,
    )
    # min_tokens=100 ensures both "# header" and "# footer" are below the
    # threshold and trigger the merge path.
    config = IndexConfig(chunk_min_tokens=100, chunk_max_tokens=500)
    chunker = ASTChunker(config)
    chunks = chunker.chunk_file(parsed, source)

    file_chunks = [c for c in chunks if c.symbol_name is None]
    # header and footer must have been merged into a single file-level chunk
    assert len(file_chunks) == 1
    assert "# header" in file_chunks[0].content
    assert "# footer" in file_chunks[0].content


def test_blank_lines_only_file_level_skipped() -> None:
    # Blank lines between two functions: no real file-level code.
    source = b"def a():\n    pass\n\n\ndef b():\n    pass\n"
    parsed = ParsedFile(
        path="two.py",
        language="python",
        symbols=[
            Symbol(
                name="a",
                qualified_name="a",
                kind=SymbolKind.FUNCTION,
                file_path="two.py",
                start_line=1,
                end_line=2,
            ),
            Symbol(
                name="b",
                qualified_name="b",
                kind=SymbolKind.FUNCTION,
                file_path="two.py",
                start_line=5,
                end_line=6,
            ),
        ],
        imports=[],
        lines=6,
    )
    chunker = _default_chunker()
    chunks = chunker.chunk_file(parsed, source)

    file_chunks = [c for c in chunks if c.symbol_name is None]
    assert file_chunks == [], f"Expected no file-level chunks, got: {file_chunks}"
