from __future__ import annotations

from pathlib import Path

import pytest

from archex.index.graph import DependencyGraph
from archex.languages import CHUNK_ONLY_LANGUAGE_IDS, LANGUAGE_SUPPORT, get_language_tier
from archex.models import Config, LanguageTier, ParsedFile, RepoMetadata
from archex.parse.adapters import LanguageAdapter, default_adapter_registry
from archex.pipeline.service import produce_artifacts
from archex.serve.profile import build_profile

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

CHUNK_ONLY_SAMPLES: dict[str, tuple[str, str, list[tuple[int, int]]]] = {
    "c": (
        "main.c",
        "#include <stdio.h>\nint add(int a, int b) { return a + b; }\nstruct Point { int x; };\n",
        [(1, 1), (2, 2), (3, 3)],
    ),
    "cpp": (
        "main.cpp",
        "#include <vector>\nnamespace n { class C { public: void m(){} }; int f(){return 1;} }\n",
        [(1, 1), (2, 2)],
    ),
    "ruby": (
        "app.rb",
        'require "json"\nmodule M\n  class C\n    def m\n    end\n  end\nend\n',
        [(1, 1), (2, 7)],
    ),
    "scala": (
        "App.scala",
        "import scala.collection.mutable\nclass C { def m(): Int = 1 }\nobject O { def f = 2 }\n",
        [(1, 1), (2, 2), (3, 3)],
    ),
    "lua": (
        "app.lua",
        'require("json")\nfunction f(x) return x end\nlocal t = { a = 1 }\n',
        [(1, 1), (2, 2), (3, 3)],
    ),
    "bash": ("run.sh", "main() { echo hi; }\nif true; then echo yes; fi\n", [(1, 1), (2, 2)]),
    "sql": (
        "schema.sql",
        "CREATE TABLE users(id INT);\n"
        "SELECT * FROM users;\n"
        "CREATE VIEW active_users AS SELECT id FROM users;\n",
        [(1, 1), (2, 2), (3, 3)],
    ),
    "html": (
        "index.html",
        "<html><body><section><h1>Hello</h1></section></body></html>\n",
        [(1, 1)],
    ),
    "xml": (
        "MoquiEntity.xml",
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<entity name="Foo" package="example">\n'
        '    <field name="id" type="id"/>\n'
        '    <field name="name" type="text-medium"/>\n'
        '    <relationship type="one" related="Bar"/>\n'
        "</entity>\n"
        '<service name="bar#Create" verb="create" noun="Bar">\n'
        "    <in-parameters>\n"
        '        <parameter name="id"/>\n'
        '        <parameter name="name"/>\n'
        "    </in-parameters>\n"
        "    <actions>\n"
        '        <service-call name="create#Bar" in-map="context"/>\n'
        "    </actions>\n"
        "</service>\n",
        [(2, 6), (7, 15)],
    ),
    "css": (
        "style.css",
        '@import url("x.css");\n.foo { color: red; }\n@media screen { .bar { display: none; } }\n',
        [(1, 1), (2, 2), (3, 3)],
    ),
    "yaml": ("config.yaml", "name: test\nitems:\n  - one\n", [(1, 3)]),
    "toml": ("config.toml", '[a]\nname = "a"\n\n[b]\nname = "b"\n', [(1, 3), (4, 5)]),
    "json": ("data.json", '{"name": "x", "items": [1, 2]}\n', [(1, 1)]),
    "markdown": ("README.md", "# Title\n\nBody\n\n## Section\n\nMore\n", [(1, 7)]),
    "solidity": (
        "C.sol",
        "pragma solidity ^0.8.0;\ncontract C { function f() public {} }\n",
        [(1, 1), (2, 2)],
    ),
}

FULL_LANGUAGE_FIXTURES: dict[str, Path] = {
    "python": FIXTURES_DIR / "python_simple",
    "typescript": FIXTURES_DIR / "typescript_simple",
    "go": FIXTURES_DIR / "go_simple",
    "rust": FIXTURES_DIR / "rust_simple",
    "java": FIXTURES_DIR / "java_simple",
    "kotlin": FIXTURES_DIR / "kotlin_simple",
    "csharp": FIXTURES_DIR / "csharp_simple",
    "swift": FIXTURES_DIR / "swift_simple",
    "php": FIXTURES_DIR / "php_simple",
}


def _adapters() -> dict[str, LanguageAdapter]:
    return default_adapter_registry.build_all()


@pytest.mark.parametrize("language_id", sorted(CHUNK_ONLY_SAMPLES))
def test_chunk_only_language_boundaries(language_id: str, tmp_path: Path) -> None:
    filename, source, expected_ranges = CHUNK_ONLY_SAMPLES[language_id]
    (tmp_path / filename).write_text(source)

    bundle = produce_artifacts(tmp_path, Config(languages=[language_id]), _adapters())

    assert len(bundle.parsed_files) == 1
    parsed = bundle.parsed_files[0]
    assert parsed.language == language_id
    assert parsed.symbols == []
    assert all(not imports for imports in bundle.resolved_imports.values())
    assert bundle.edges == []
    assert [(item.start_line, item.end_line) for item in parsed.chunk_ranges] == expected_ranges
    chunk_ranges = [(chunk.start_line, chunk.end_line) for chunk in bundle.chunks]
    for expected_range in expected_ranges:
        assert expected_range in chunk_ranges
    assert all(chunk.symbol_name is None for chunk in bundle.chunks)


@pytest.mark.parametrize("language_id", sorted(FULL_LANGUAGE_FIXTURES))
def test_full_tier_languages_extract_symbols_and_imports(language_id: str) -> None:
    bundle = produce_artifacts(
        FULL_LANGUAGE_FIXTURES[language_id],
        Config(languages=[language_id]),
        _adapters(),
    )

    assert get_language_tier(language_id) == LanguageTier.FULL
    assert sum(len(parsed.symbols) for parsed in bundle.parsed_files) > 0
    assert any(imports for imports in bundle.resolved_imports.values())


@pytest.mark.parametrize("language_id", sorted(CHUNK_ONLY_LANGUAGE_IDS))
def test_chunk_only_languages_report_chunk_only_tier(language_id: str) -> None:
    assert get_language_tier(language_id) == LanguageTier.CHUNK_ONLY


def test_javascript_full_tier_extracts_symbols_and_imports(tmp_path: Path) -> None:
    (tmp_path / "dep.js").write_text("export const value = 1;\n")
    (tmp_path / "app.js").write_text(
        "import { value } from './dep.js';\nexport function run() { return value; }\n"
    )

    bundle = produce_artifacts(tmp_path, Config(languages=["javascript"]), _adapters())

    assert get_language_tier("javascript") == LanguageTier.FULL
    assert sum(len(parsed.symbols) for parsed in bundle.parsed_files) > 0
    assert any(imports for imports in bundle.resolved_imports.values())


def test_language_stats_include_tier_labels() -> None:
    parsed_files = [
        ParsedFile(path="main.py", language="python", lines=10),
        ParsedFile(path="schema.sql", language="sql", lines=3),
    ]
    graph = DependencyGraph.from_parsed_files(parsed_files, {})
    profile = build_profile(
        RepoMetadata(commit_hash="abc123", total_files=2, total_lines=13),
        parsed_files,
        graph,
    )

    assert profile.stats.languages["python"].tier == LanguageTier.FULL
    assert profile.stats.languages["sql"].tier == LanguageTier.CHUNK_ONLY


def test_language_registry_has_required_coverage() -> None:
    required = set(CHUNK_ONLY_SAMPLES) | set(FULL_LANGUAGE_FIXTURES) | {"javascript", "tsx"}
    assert required <= set(LANGUAGE_SUPPORT)
    assert len(LANGUAGE_SUPPORT) >= 25
