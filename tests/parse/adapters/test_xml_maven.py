"""Tests for the Maven POM XML dialect plugin (M14).

`is_maven_pom` (src/archex/parse/adapters/xml_maven.py) gates every other
function this dialect plugin adds: a file only ever gets Maven-specific
treatment when it is named `pom.xml` *and* its root element is
`<project>`. Both signals are required independently -- Apache Ant's
`build.xml` also roots on `<project>` but is not named `pom.xml`, and a
file that happens to be named `pom.xml` but is not a real Maven
descriptor must not be treated as one either. The false-positive checks
below run detection directly against M13's generic-XML fixtures
(`tests/fixtures/xml_structured/`) to confirm the dialect plugin never
claims ordinary XML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.models import ImportStatement
from archex.parse.adapters.xml_maven import (
    extract_maven_dependencies,
    is_maven_pom,
    resolve_maven_dependency,
)
from archex.parse.engine import TreeSitterEngine

FIXTURES_DIR = "tests/fixtures/xml_maven"
GENERIC_XML_FIXTURES_DIR = "tests/fixtures/xml_structured"


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


def parse(engine: TreeSitterEngine, source: bytes) -> object:
    return engine.parse_bytes(source, "xml")


def _read(path: str) -> bytes:
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# is_maven_pom: positive detection across the multi-module fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative_path",
    [
        "pom.xml",
        "module-a/pom.xml",
        "module-b/pom.xml",
        "module-c/pom.xml",
    ],
)
def test_is_maven_pom_detects_every_pom_in_the_multi_module_fixture(
    engine: TreeSitterEngine, relative_path: str
) -> None:
    source = _read(f"{FIXTURES_DIR}/{relative_path}")

    assert is_maven_pom(relative_path, parse(engine, source), source) is True


# ---------------------------------------------------------------------------
# is_maven_pom: false-positive guards
# ---------------------------------------------------------------------------


def test_is_maven_pom_rejects_ant_build_xml_despite_matching_project_root(
    engine: TreeSitterEngine,
) -> None:
    """Apache Ant's `build.xml` also roots on `<project>` -- filename must
    still gate detection even when the root-element signal matches."""
    source = _read(f"{FIXTURES_DIR}/build.xml")

    assert is_maven_pom("build.xml", parse(engine, source), source) is False


@pytest.mark.parametrize("filename", ["catalog.xml", "library.xml"])
def test_is_maven_pom_rejects_m13_generic_xml_fixtures_by_filename(
    engine: TreeSitterEngine, filename: str
) -> None:
    """Neither M13 generic-XML fixture is named `pom.xml`, so filename
    alone must reject both regardless of root-element shape."""
    source = _read(f"{GENERIC_XML_FIXTURES_DIR}/{filename}")

    assert is_maven_pom(filename, parse(engine, source), source) is False


@pytest.mark.parametrize("filename", ["catalog.xml", "library.xml"])
def test_is_maven_pom_rejects_m13_generic_xml_content_even_when_renamed_to_pom_xml(
    engine: TreeSitterEngine, filename: str
) -> None:
    """Isolates the root-element signal: even if one of M13's fixtures
    were renamed to `pom.xml`, neither `<catalog>` nor `<library>` is a
    `<project>` root, so detection must still reject them."""
    source = _read(f"{GENERIC_XML_FIXTURES_DIR}/{filename}")

    assert is_maven_pom("pom.xml", parse(engine, source), source) is False


def test_is_maven_pom_rejects_pom_xml_named_file_with_non_project_root(
    engine: TreeSitterEngine,
) -> None:
    """A file literally named `pom.xml` whose root element is not
    `<project>` (e.g. mis-generated tooling output) is not a Maven POM."""
    source = b"<config><setting>value</setting></config>"

    assert is_maven_pom("pom.xml", parse(engine, source), source) is False


# ---------------------------------------------------------------------------
# extract_maven_dependencies: groupId/artifactId/version extraction
# ---------------------------------------------------------------------------


def test_extract_maven_dependencies_captures_group_artifact_version(
    engine: TreeSitterEngine,
) -> None:
    source = _read(f"{FIXTURES_DIR}/module-b/pom.xml")

    references = extract_maven_dependencies(parse(engine, source), source, "module-b/pom.xml")

    assert [ref.module for ref in references] == ["com.example:module-a:1.0.0"]
    assert all(ref.file_path == "module-b/pom.xml" for ref in references)
    assert all(ref.is_relative is False for ref in references)


def test_extract_maven_dependencies_captures_every_declared_dependency_including_external(
    engine: TreeSitterEngine,
) -> None:
    """module-c depends on module-b (intra-repo) and junit (external) --
    both are extracted here; whether one stays external is a
    `resolve_maven_dependency` concern, not extraction's."""
    source = _read(f"{FIXTURES_DIR}/module-c/pom.xml")

    references = extract_maven_dependencies(parse(engine, source), source, "module-c/pom.xml")

    assert {ref.module for ref in references} == {
        "com.example:module-b:1.0.0",
        "junit:junit:4.13.2",
    }


def test_extract_maven_dependencies_ignores_parent_declaration(engine: TreeSitterEngine) -> None:
    """`<parent>` carries its own groupId/artifactId/version but is not a
    `<dependency>` -- module-a has no `<dependencies>` block at all, only
    a `<parent>` reference back to the aggregator, and must yield zero
    references."""
    source = _read(f"{FIXTURES_DIR}/module-a/pom.xml")

    assert extract_maven_dependencies(parse(engine, source), source, "module-a/pom.xml") == []


def test_extract_maven_dependencies_ignores_dependency_management_block(
    engine: TreeSitterEngine,
) -> None:
    """A `<dependencyManagement><dependencies><dependency>` entry is
    version-pinning, not a declared usage edge -- it lives two levels
    below the root, never as a direct child `<dependencies>` block."""
    source = b"""<?xml version="1.0" encoding="UTF-8"?>
<project>
  <groupId>com.example</groupId>
  <artifactId>bom-consumer</artifactId>
  <version>1.0.0</version>
  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>com.example</groupId>
        <artifactId>should-not-be-extracted</artifactId>
        <version>1.0.0</version>
      </dependency>
    </dependencies>
  </dependencyManagement>
</project>
"""

    assert extract_maven_dependencies(parse(engine, source), source, "pom.xml") == []


def test_extract_maven_dependencies_returns_empty_for_non_pom_root(
    engine: TreeSitterEngine,
) -> None:
    source = b"<config><dependencies><dependency>ignored</dependency></dependencies></config>"

    assert extract_maven_dependencies(parse(engine, source), source, "pom.xml") == []


# ---------------------------------------------------------------------------
# resolve_maven_dependency: artifactId-directory convention
# ---------------------------------------------------------------------------


def _file_map() -> dict[str, str]:
    return {
        "pom": "pom.xml",
        "module-a.pom": "module-a/pom.xml",
        "module-b.pom": "module-b/pom.xml",
        "module-c.pom": "module-c/pom.xml",
    }


def test_resolve_maven_dependency_resolves_sibling_module_by_artifact_directory() -> None:
    imp = ImportStatement(
        module="com.example:module-a:1.0.0", file_path="module-b/pom.xml", line=10
    )

    assert resolve_maven_dependency(imp, _file_map()) == "module-a/pom.xml"


def test_resolve_maven_dependency_returns_none_for_external_coordinate() -> None:
    imp = ImportStatement(module="junit:junit:4.13.2", file_path="module-c/pom.xml", line=20)

    assert resolve_maven_dependency(imp, _file_map()) is None


def test_resolve_maven_dependency_returns_none_when_target_directory_is_ambiguous() -> None:
    file_map = {**_file_map(), "vendored.module-a.pom": "third_party/module-a/pom.xml"}
    imp = ImportStatement(
        module="com.example:module-a:1.0.0", file_path="module-b/pom.xml", line=10
    )

    assert resolve_maven_dependency(imp, file_map) is None


def test_resolve_maven_dependency_returns_none_for_self_reference() -> None:
    imp = ImportStatement(module="com.example:module-a:1.0.0", file_path="module-a/pom.xml", line=5)

    assert resolve_maven_dependency(imp, _file_map()) is None


def test_resolve_maven_dependency_returns_none_for_malformed_coordinate() -> None:
    imp = ImportStatement(module="not-a-coordinate", file_path="module-b/pom.xml", line=1)

    assert resolve_maven_dependency(imp, _file_map()) is None
