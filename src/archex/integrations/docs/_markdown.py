"""Shared local-markdown-link extraction used by the doc_link and adr providers.

Internal helper module: never imported outside ``archex.integrations.docs``.
Extraction is purely local-text regex matching against files already on
disk -- no network access, no remote link resolution, and no fabrication of
a relation to a path that does not exist under the repository root.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Matches ``[text](target)`` markdown links. Deliberately does not match
#: reference-style links (``[text][ref]``) or bare autolinks -- the common,
#: unambiguous case is sufficient local evidence without a full CommonMark
#: parser dependency.
_LINK_PATTERN = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")

#: Bounds how many candidate links one document contributes, keeping a
#: pathological document (a huge generated index page) from dominating a
#: run's evidence set.
MAX_LINKS_PER_DOCUMENT = 200


def extract_markdown_links(doc_path: Path, repo_root: Path) -> list[tuple[str, str]]:
    """Return ``(link_text, target_path)`` pairs for links in *doc_path* that
    resolve to a real file under *repo_root*.

    ``target_path`` is repo-root-relative with POSIX separators. Remote
    URLs, mailto links, and in-page anchors are never resolved or recorded.
    A link resolved relative to *doc_path*'s own directory first, falling
    back to resolution relative to *repo_root* (both are conventional
    markdown-link resolution bases); any target escaping *repo_root* is
    discarded.
    """
    try:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    results: list[tuple[str, str]] = []
    for link_text, raw_target in _LINK_PATTERN.findall(text)[:MAX_LINKS_PER_DOCUMENT]:
        resolved = _resolve_link_target(raw_target, doc_path=doc_path, repo_root=repo_root)
        if resolved is not None:
            results.append((link_text.strip(), resolved))
    return results


def _resolve_link_target(raw_target: str, *, doc_path: Path, repo_root: Path) -> str | None:
    target = raw_target.split("#", maxsplit=1)[0].strip()
    if not target or "://" in target or target.startswith(("mailto:", "#")):
        return None

    for base in (doc_path.parent, repo_root):
        candidate = (base / target).resolve()
        try:
            relative = candidate.relative_to(repo_root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return relative.as_posix()
    return None
