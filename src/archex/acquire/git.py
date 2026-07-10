"""Git-based repository acquisition: clone, sparse-checkout, and commit pinning."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from archex.exceptions import AcquireError

_ALLOWED_SCHEMES = ("http", "https")

# Matches any RFC 3986 `scheme://...` URL (ssh://, git://, file://, ftp(s)://,
# ...), capturing the scheme name. Scheme names are case-insensitive per
# RFC 3986 and real git/libcurl honor that, so the captured group is
# compared case-insensitively below rather than via a case-sensitive prefix
# check.
_URL_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://")

# Matches git's `<transport>::<address>` remote-helper syntax. `ext::` in
# particular runs its address as an arbitrary shell command — a well-known
# RCE vector when an unvalidated URL reaches `git clone` (e.g. "ext::sh -c
# 'touch pwned'"). No archex feature needs any remote helper, so all of them
# are rejected rather than enumerated.
_TRANSPORT_HELPER_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*::")

# A Windows absolute drive path ("C:\Users\x", "C:/Users/x") is a single
# ASCII letter immediately followed by `:` and a path separator. git itself
# special-cases exactly this shape to disambiguate it from the scp-like
# shorthand below; without this, _SCP_LIKE_RE would reject every Windows
# absolute path as if it were an SSH host named "C".
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

# Matches the scp-like ssh shorthand `[user@]host:path` (no scheme, no
# `//`), e.g. "git@github.com:user/repo.git" or, just as validly for git,
# the user-less "github.com:user/repo.git". The `user@` part is OPTIONAL in
# git's real grammar — a regex requiring it (as an earlier version of this
# function did) lets a bare `host:path` slip through as "a local path" and
# straight into `git clone`, resulting in a real outbound SSH connection to
# an attacker-chosen host.
_SCP_LIKE_RE = re.compile(r"^(?:[^/@\s]+@)?[^/\s:]+:")

_BRANCH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")


def is_remote_url(url: str) -> bool:
    """True if url is an http(s) URL that clone_repo() should fetch remotely.

    Case-insensitive scheme match — matches validate_url()'s own handling.
    Every caller that decides "clone this vs. treat it as a local path"
    (_acquire(), resolve_source()) must agree with validate_url() on this
    exact question, or validate_url()'s hardening never reaches the real
    call path: a string that fails a separately re-implemented, stricter
    gate upstream never reaches validate_url() at all.
    """
    scheme_match = _URL_SCHEME_RE.match(url)
    return scheme_match is not None and scheme_match.group(1).lower() in _ALLOWED_SCHEMES


def validate_url(url: str) -> None:
    """Raise AcquireError unless url is http(s) or a genuine local filesystem path.

    Uses an allowlist, not a denylist: anything that isn't explicitly
    http://, https://, or scheme-free is rejected, rather than only the
    specific schemes this function happens to enumerate. That covers every
    URL-scheme form (`scheme://...`), git's `transport::address`
    remote-helper syntax, and the scp-like `[user@]host:path` shorthand in
    one pass, so a crafted RepoSource cannot reach a transport this function
    hasn't seen before.
    """
    if is_remote_url(url):
        return
    if _WINDOWS_DRIVE_RE.match(url):
        return
    if _URL_SCHEME_RE.match(url) or _TRANSPORT_HELPER_RE.match(url) or _SCP_LIKE_RE.match(url):
        raise AcquireError(
            f"Disallowed URL scheme in {url!r}: only http://, https://, and local paths are allowed"
        )
    # Anything else is treated as a local filesystem path.


def validate_branch(branch: str) -> None:
    """Raise AcquireError if branch name is unsafe."""
    if not _BRANCH_RE.match(branch):
        raise AcquireError(
            f"Invalid branch name {branch!r}: must match ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$"
        )


def clone_repo(
    url: str,
    target_dir: str | Path,
    shallow: bool = True,
    branch: str | None = None,
) -> Path:
    """Clone a git repository to target_dir and return the resolved path.

    Raises AcquireError on subprocess failure or timeout.
    """
    validate_url(url)
    if branch is not None:
        validate_branch(branch)

    target = Path(target_dir).resolve()
    cmd: list[str] = ["git", "clone"]

    if shallow:
        cmd += ["--depth", "1"]

    if branch is not None:
        cmd += ["--branch", branch]

    cmd += [url, str(target)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        raise AcquireError(f"git clone failed for {url!r}: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AcquireError(f"git clone timed out after 120s for {url!r}") from exc

    return target
