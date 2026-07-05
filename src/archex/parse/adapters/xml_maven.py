"""Maven POM XML dialect plugin.

Layered on top of the generic `XmlAdapter` (M13): detects `pom.xml` files
by filename *and* root element, then extracts Maven's `<dependency>`
declarations as cross-module references.

Detection requires both signals -- filename and root element -- because
either alone is ambiguous: Apache Ant's `build.xml` also roots on
`<project>`, and a file that merely happens to be named `pom.xml` is not
guaranteed to be a real Maven descriptor (e.g. a fixture or unrelated
config reusing the name).

`<parent>` inheritance and `<dependencyManagement>`/`<profiles>`-scoped
dependency lists are intentionally excluded from extraction: only
`<dependency>` elements that are direct children of a `<dependencies>`
element that is itself a direct child of the `<project>` root count as
declared dependencies. Resolution of the extracted coordinates to repo
paths is added separately (see the M14 dependency-edge milestone).
"""

from __future__ import annotations

import posixpath
from typing import Any

from archex.models import ImportStatement

_POM_FILENAME = "pom.xml"
_POM_ROOT_TAG = "project"
_DEPENDENCIES_TAG = "dependencies"
_DEPENDENCY_TAG = "dependency"


def is_maven_pom(file_path: str, tree: object, source: bytes) -> bool:
    """True when `file_path` is a real Maven POM: named `pom.xml` with a
    `<project>` root element. Neither signal alone is sufficient -- see
    module docstring."""
    if posixpath.basename(file_path.replace("\\", "/")) != _POM_FILENAME:
        return False
    root = _root_element(tree)
    if root is None:
        return False
    return _element_tag_name(root, source) == _POM_ROOT_TAG


def extract_maven_dependencies(
    tree: object, source: bytes, file_path: str
) -> list[ImportStatement]:
    """Extract `<dependency>` declarations from the project's direct
    `<dependencies>` block. `<parent>` and any `<dependencyManagement>`- or
    `<profiles>`-nested dependency lists are not direct children of the
    root and are therefore never visited -- see module docstring."""
    root = _root_element(tree)
    if root is None or _element_tag_name(root, source) != _POM_ROOT_TAG:
        return []

    dependencies_el = _named_child_element(root, _DEPENDENCIES_TAG, source)
    if dependencies_el is None:
        return []

    references: list[ImportStatement] = []
    for dependency_el in _named_child_elements(dependencies_el, _DEPENDENCY_TAG, source):
        group_id = _child_text(dependency_el, "groupId", source)
        artifact_id = _child_text(dependency_el, "artifactId", source)
        if group_id is None or artifact_id is None:
            continue
        version = _child_text(dependency_el, "version", source)
        coordinate = (
            f"{group_id}:{artifact_id}:{version}" if version else f"{group_id}:{artifact_id}"
        )
        references.append(
            ImportStatement(
                module=coordinate,
                file_path=file_path,
                line=int(dependency_el.start_point[0]) + 1,
                is_relative=False,
            )
        )
    return references


# ---------------------------------------------------------------------------
# tree-sitter-xml node helpers
# ---------------------------------------------------------------------------


def _root_element(tree: object) -> Any | None:
    parsed_tree: Any = tree
    for child in parsed_tree.root_node.children:
        if child.type == "element":
            return child
    return None


def _element_tag_name(element: Any, source: bytes) -> str | None:
    for child in element.children:
        if child.type in ("STag", "EmptyElemTag"):
            for grandchild in child.children:
                if grandchild.type == "Name":
                    return _node_text(grandchild, source)
    return None


def _content_child(element: Any) -> Any | None:
    for child in element.children:
        if child.type == "content":
            return child
    return None


def _direct_child_elements(element: Any) -> list[Any]:
    content = _content_child(element)
    if content is None:
        return []
    return [child for child in content.children if child.type == "element"]


def _named_child_element(element: Any, name: str, source: bytes) -> Any | None:
    for child in _direct_child_elements(element):
        if _element_tag_name(child, source) == name:
            return child
    return None


def _named_child_elements(element: Any, name: str, source: bytes) -> list[Any]:
    return [
        child
        for child in _direct_child_elements(element)
        if _element_tag_name(child, source) == name
    ]


def _element_text(element: Any, source: bytes) -> str | None:
    content = _content_child(element)
    if content is None:
        return None
    text = "".join(
        _node_text(child, source) for child in content.children if child.type == "CharData"
    ).strip()
    return text or None


def _child_text(element: Any, name: str, source: bytes) -> str | None:
    child = _named_child_element(element, name, source)
    if child is None:
        return None
    return _element_text(child, source)


def _node_text(node: Any, source: bytes) -> str:
    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")
