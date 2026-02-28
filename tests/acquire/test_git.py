from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from archex.acquire.git import clone_repo, validate_branch, validate_url
from archex.exceptions import AcquireError

if TYPE_CHECKING:
    from pathlib import Path


def test_clone_success(tmp_path: Path) -> None:
    target = tmp_path / "cloned"
    with patch("archex.acquire.git.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = clone_repo("https://example.com/repo.git", target)
    assert result == target.resolve()
    args = mock_run.call_args[0][0]
    assert "git" in args
    assert "clone" in args
    assert str(target.resolve()) in args


def test_clone_shallow_flag(tmp_path: Path) -> None:
    target = tmp_path / "shallow"
    with patch("archex.acquire.git.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        clone_repo("https://example.com/repo.git", target, shallow=True)
    args = mock_run.call_args[0][0]
    assert "--depth" in args
    assert "1" in args


def test_clone_no_shallow_flag(tmp_path: Path) -> None:
    target = tmp_path / "full"
    with patch("archex.acquire.git.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        clone_repo("https://example.com/repo.git", target, shallow=False)
    args = mock_run.call_args[0][0]
    assert "--depth" not in args


def test_invalid_url_raises(tmp_path: Path) -> None:
    target = tmp_path / "bad"
    with patch("archex.acquire.git.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            128, "git clone", stderr=b"repository not found"
        )
        with pytest.raises(AcquireError, match="git clone failed"):
            clone_repo("https://invalid.example/no-repo.git", target)


def test_timeout_raises(tmp_path: Path) -> None:
    target = tmp_path / "timeout"
    with patch("archex.acquire.git.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("git clone", 120)
        with pytest.raises(AcquireError, match="timed out"):
            clone_repo("https://example.com/slow.git", target)


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/user/repo.git",
        "http://example.com/repo.git",
        "/home/user/local-repo",
        "./relative/path",
    ],
)
def testvalidate_url_accepts_safe_urls(url: str) -> None:
    validate_url(url)  # must not raise


@pytest.mark.parametrize(
    "bad_url",
    [
        "git@github.com:user/repo.git",
        "ssh://git@github.com/user/repo.git",
        "file:///etc/passwd",
        "git://github.com/user/repo.git",
        "ftp://example.com/repo.git",
    ],
)
def testvalidate_url_rejects_disallowed_schemes(bad_url: str) -> None:
    with pytest.raises(AcquireError, match="Disallowed URL scheme"):
        validate_url(bad_url)


def test_clone_repo_rejects_ssh_url(tmp_path: Path) -> None:
    with pytest.raises(AcquireError, match="Disallowed URL scheme"):
        clone_repo("git@github.com:user/repo.git", tmp_path / "out")


def test_clone_repo_rejects_file_url(tmp_path: Path) -> None:
    with pytest.raises(AcquireError, match="Disallowed URL scheme"):
        clone_repo("file:///etc/passwd", tmp_path / "out")


# ---------------------------------------------------------------------------
# Branch validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "feature/my-branch",
        "release/v1.0.0",
        "fix.bug.123",
        "abc123",
    ],
)
def testvalidate_branch_accepts_safe_names(branch: str) -> None:
    validate_branch(branch)  # must not raise


@pytest.mark.parametrize(
    "bad_branch",
    [
        "-c core.sshCommand=evil",
        "--upload-pack=evil",
        "; rm -rf /",
        "$(whoami)",
        "`id`",
        "",
        " ",
        "-main",  # starts with dash
    ],
)
def testvalidate_branch_rejects_unsafe_names(bad_branch: str) -> None:
    with pytest.raises(AcquireError, match="Invalid branch name"):
        validate_branch(bad_branch)


def test_clone_repo_rejects_malicious_branch(tmp_path: Path) -> None:
    with pytest.raises(AcquireError, match="Invalid branch name"):
        clone_repo(
            "https://example.com/repo.git",
            tmp_path / "out",
            branch="-c core.sshCommand=evil",
        )


def test_clone_repo_rejects_upload_pack_injection(tmp_path: Path) -> None:
    with pytest.raises(AcquireError, match="Invalid branch name"):
        clone_repo(
            "https://example.com/repo.git",
            tmp_path / "out",
            branch="--upload-pack=evil",
        )


@pytest.mark.network
def test_clone_real_repo(tmp_path: Path) -> None:
    target = tmp_path / "real"
    result = clone_repo(
        "https://github.com/pallets/click.git",
        target,
        shallow=True,
    )
    assert result.exists()
    assert (result / ".git").exists()
