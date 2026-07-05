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

from archex.parse.adapters.xml_maven import is_maven_pom
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
