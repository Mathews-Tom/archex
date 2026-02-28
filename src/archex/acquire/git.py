"""Git-based repository acquisition: clone, sparse-checkout, and commit pinning."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from archex.exceptions import AcquireError

_ALLOWED_URL_PREFIXES = ("http://", "https://")
_DISALLOWED_URL_PREFIXES = ("git@", "ssh://", "file://", "git://", "ftp://", "ftps://")
_BRANCH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*$")


def validate_url(url: str) -> None:
    """Raise AcquireError if url is not http/https or a local filesystem path."""
    if url.startswith(_ALLOWED_URL_PREFIXES):
        return
    for prefix in _DISALLOWED_URL_PREFIXES:
        if url.startswith(prefix):
            raise AcquireError(
                f"Disallowed URL scheme in {url!r}: "
                "only http://, https://, and local paths are allowed"
            )
    # Treat anything else as a local path — acceptable


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
