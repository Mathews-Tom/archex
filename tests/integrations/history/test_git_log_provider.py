"""Tests for the git-log history evidence provider."""

from __future__ import annotations

import subprocess
from pathlib import Path

from archex.integrations.history.git_log_provider import GitLogHistoryProvider
from archex.integrations.history.models import ProviderAvailability


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class TestGitLogHistoryProviderProbe:
    def test_unavailable_for_non_git_directory(self, tmp_path: Path) -> None:
        provider = GitLogHistoryProvider()
        receipt = provider.probe(tmp_path, expected_revision="HEAD")
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_available_for_real_repo(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, {"a.py": "x = 1\n"}, "add a")
        provider = GitLogHistoryProvider()
        receipt = provider.probe(repo, expected_revision="HEAD")
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.observed_revision == _head(repo)

    def test_unavailable_for_unresolvable_revision(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, {"a.py": "x = 1\n"}, "add a")
        provider = GitLogHistoryProvider()
        receipt = provider.probe(repo, expected_revision="deadbeef00000000000000000000000000000000")
        assert receipt.availability == ProviderAvailability.UNAVAILABLE


class TestGitLogHistoryProviderCollect:
    def test_collect_returns_empty_for_non_git_directory(self, tmp_path: Path) -> None:
        provider = GitLogHistoryProvider()
        cards, coupling, receipt = provider.collect(
            tmp_path, expected_revision="HEAD", max_commits=100
        )
        assert cards == []
        assert coupling == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_collect_produces_change_cards(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, {"a.py": "x = 1\n"}, "add a (#42)")
        _commit(repo, {"a.py": "x = 2\n", "b.py": "y = 1\n"}, "update a, add b")

        provider = GitLogHistoryProvider()
        cards, _coupling, receipt = provider.collect(
            repo, expected_revision="HEAD", max_commits=100
        )
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.window_commit_count == 2
        assert len(cards) == 2
        first_commit_card = next(c for c in cards if "#42" in c.commit_subject)
        assert first_commit_card.changed_files == ["a.py"]
        assert first_commit_card.linked_references[0].identifier == "42"

    def test_collect_bounds_window_by_max_commits(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        for i in range(5):
            _commit(repo, {f"f{i}.py": "x = 1\n"}, f"add f{i}")

        provider = GitLogHistoryProvider()
        cards, _coupling, receipt = provider.collect(repo, expected_revision="HEAD", max_commits=3)
        assert receipt.window_commit_count == 3
        assert len(cards) == 3

    def test_collect_identifies_touched_test_files(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(
            repo,
            {"src/feature.py": "x = 1\n", "tests/test_feature.py": "def test_x(): pass\n"},
            "add feature with test",
        )

        provider = GitLogHistoryProvider()
        cards, _coupling, _receipt = provider.collect(
            repo, expected_revision="HEAD", max_commits=100
        )
        assert cards[0].touched_test_files == ["tests/test_feature.py"]

    def test_collect_produces_temporal_coupling_for_repeated_co_change(
        self, tmp_path: Path
    ) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, {"a.py": "1", "b.py": "1"}, "init")
        _commit(repo, {"a.py": "2", "b.py": "2"}, "co-change 1")
        _commit(repo, {"a.py": "3", "b.py": "3"}, "co-change 2")

        provider = GitLogHistoryProvider()
        _cards, coupling, receipt = provider.collect(
            repo, expected_revision="HEAD", max_commits=100
        )
        assert receipt.window_commit_count == 3
        assert len(coupling) == 1
        pair = coupling[0]
        assert {pair.file_a, pair.file_b} == {"a.py", "b.py"}
        assert pair.co_change_count == 3
        assert pair.window_commit_count == 3

    def test_collect_omits_single_co_change_pairs(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        _commit(repo, {"a.py": "1", "b.py": "1"}, "only co-change once")

        provider = GitLogHistoryProvider()
        _cards, coupling, _receipt = provider.collect(
            repo, expected_revision="HEAD", max_commits=100
        )
        assert coupling == []

    def test_collect_excludes_dense_commits_from_coupling(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        many_files = {f"f{i}.py": "x" for i in range(40)}
        _commit(repo, many_files, "mass commit")
        many_files_v2 = {f"f{i}.py": "y" for i in range(40)}
        _commit(repo, many_files_v2, "mass commit again")

        provider = GitLogHistoryProvider()
        _cards, coupling, _receipt = provider.collect(
            repo, expected_revision="HEAD", max_commits=100
        )
        assert coupling == []
