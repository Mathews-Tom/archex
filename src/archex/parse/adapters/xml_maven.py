"""Maven POM XML dialect plugin.

Layered on top of the generic `XmlAdapter` (M13): detects `pom.xml` files
by filename *and* root element. Dependency extraction and resolution are
added on top of this detector separately (see the M14 dependency-edge
milestone).

Detection requires both signals -- filename and root element -- because
either alone is ambiguous: Apache Ant's `build.xml` also roots on
`<project>`, and a file that merely happens to be named `pom.xml` is not
guaranteed to be a real Maven descriptor (e.g. a fixture or unrelated
config reusing the name).
"""

from __future__ import annotations

import posixpath
from typing import Any

_POM_FILENAME = "pom.xml"
_POM_ROOT_TAG = "project"


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


def _node_text(node: Any, source: bytes) -> str:
    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")
