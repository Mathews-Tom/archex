"""Tests for the recency/churn ranking prior provider."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from archex.index.churn import (
    CHURN_FIXTURE_SCHEMA,
    CHURN_SOURCE_FIXTURE,
    CHURN_SOURCE_HISTORY,
    CHURN_SOURCE_NEUTRAL,
    DEFAULT_CHURN_MAX_BOOST,
    ChurnError,
    load_churn_priors,
)

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "churn" / "sample_churn.json"


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    import os

    env = os.environ.copy()
    if date is not None:
        env.update(
            {
                "GIT_AUTHOR_DATE": date,
                "GIT_COMMITTER_DATE": date,
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@example.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@example.com",
            }
        )
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")


def _commit(repo: Path, path: str, content: str, *, date: str) -> None:
    (repo / path).write_text(content, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", f"touch {path} @ {date}", date=date)


def _init_history_repo(repo: Path) -> None:
    """A small repo where ``hot.py`` is frequently and recently changed and
    ``cold.py`` is touched once long ago."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _commit(repo, "cold.py", "v1\n", date="2020-01-01T00:00:00")
    _commit(repo, "warm.py", "v1\n", date="2020-02-01T00:00:00")
    _commit(repo, "hot.py", "v1\n", date="2020-03-01T00:00:00")
    _commit(repo, "warm.py", "v2\n", date="2021-01-01T00:00:00")
    _commit(repo, "hot.py", "v2\n", date="2022-01-01T00:00:00")
    _commit(repo, "hot.py", "v3\n", date="2023-01-01T00:00:00")
    _commit(repo, "hot.py", "v4\n", date="2024-01-01T00:00:00")


class TestFixtureSource:
    def test_fixture_parses_and_is_stable(self) -> None:
        first = load_churn_priors(_FIXTURE.parent, fixture_path=_FIXTURE)
        second = load_churn_priors(_FIXTURE.parent, fixture_path=_FIXTURE)

        assert first.source == CHURN_SOURCE_FIXTURE
        assert first.commit == "0fbd93c0000000000000000000000000000000aa"
        # Reproducible: the same fixture yields the same priors every load.
        assert first.priors == second.priors

    def test_fixture_priors_are_ordered_and_bounded(self) -> None:
        priors = load_churn_priors(_FIXTURE.parent, fixture_path=_FIXTURE)
        ceiling = 1.0 + DEFAULT_CHURN_MAX_BOOST

        hot = priors.prior_for("src/app/hot.py")
        warm = priors.prior_for("src/app/warm.py")
        cold = priors.prior_for("src/app/cold.py")
        stable = priors.prior_for("src/app/stable_core.py")

        # More churn + more recency => larger (but still bounded) multiplier.
        assert hot > warm > stable > cold > 1.0
        for multiplier in (hot, warm, stable, cold):
            assert 1.0 < multiplier <= ceiling
        # The most-churned file saturates exactly at the boost ceiling.
        assert abs(hot - ceiling) < 1e-9

    def test_unknown_file_is_neutral(self) -> None:
        priors = load_churn_priors(_FIXTURE.parent, fixture_path=_FIXTURE)
        assert priors.prior_for("src/app/never_seen.py") == 1.0

    def test_zero_max_boost_yields_no_priors(self) -> None:
        priors = load_churn_priors(_FIXTURE.parent, fixture_path=_FIXTURE, max_boost=0.0)
        assert priors.source == CHURN_SOURCE_FIXTURE
        assert priors.priors == {}
        assert priors.prior_for("src/app/hot.py") == 1.0


class TestFixtureValidation:
    def test_bad_schema_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "churn.json"
        bad.write_text('{"schema": "nope", "files": {}}', encoding="utf-8")
        with pytest.raises(ChurnError, match="schema"):
            load_churn_priors(tmp_path, fixture_path=bad)

    def test_missing_files_object_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "churn.json"
        bad.write_text(f'{{"schema": "{CHURN_FIXTURE_SCHEMA}"}}', encoding="utf-8")
        with pytest.raises(ChurnError, match="files"):
            load_churn_priors(tmp_path, fixture_path=bad)

    def test_invalid_commit_count_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "churn.json"
        bad.write_text(
            f'{{"schema": "{CHURN_FIXTURE_SCHEMA}", '
            '"files": {"a.py": {"commits": -1, "recency": 0.5}}}',
            encoding="utf-8",
        )
        with pytest.raises(ChurnError, match="commits"):
            load_churn_priors(tmp_path, fixture_path=bad)

    def test_non_finite_recency_raises(self, tmp_path: Path) -> None:
        # json.loads accepts the non-standard NaN token; it must be rejected so a
        # malformed fixture cannot smuggle a NaN multiplier into scoring.
        bad = tmp_path / "churn.json"
        bad.write_text(
            f'{{"schema": "{CHURN_FIXTURE_SCHEMA}", '
            '"files": {"a.py": {"commits": 3, "recency": NaN}}}',
            encoding="utf-8",
        )
        with pytest.raises(ChurnError, match="recency"):
            load_churn_priors(tmp_path, fixture_path=bad)


class TestNeutralFallback:
    def test_missing_history_returns_neutral(self, tmp_path: Path) -> None:
        # No .git and no fixture => neutral prior for every file.
        priors = load_churn_priors(tmp_path)
        assert priors.source == CHURN_SOURCE_NEUTRAL
        assert priors.priors == {}
        assert priors.prior_for("anything.py") == 1.0

    def test_single_commit_repo_returns_neutral(self, tmp_path: Path) -> None:
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "config", "user.email", "test@example.com")
        _git(tmp_path, "config", "user.name", "Test")
        _commit(tmp_path, "only.py", "v1\n", date="2020-01-01T00:00:00")
        priors = load_churn_priors(tmp_path)
        # A single commit carries no churn signal => neutral.
        assert priors.source == CHURN_SOURCE_NEUTRAL
        assert priors.priors == {}


class TestHistorySource:
    def test_history_backed_churn_is_deterministic(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_history_repo(repo)

        first = load_churn_priors(repo)
        second = load_churn_priors(repo)

        assert first.source == CHURN_SOURCE_HISTORY
        assert first.commit != ""
        # Reproducible given the fixed commit graph.
        assert first.priors == second.priors

    def test_history_orders_by_churn_and_recency(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _init_history_repo(repo)
        priors = load_churn_priors(repo)
        ceiling = 1.0 + DEFAULT_CHURN_MAX_BOOST

        hot = priors.prior_for("hot.py")
        warm = priors.prior_for("warm.py")
        cold = priors.prior_for("cold.py")

        assert hot > warm > cold
        for multiplier in (hot, warm, cold):
            assert 1.0 <= multiplier <= ceiling

    def test_shallow_clone_yields_neutral(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _init_history_repo(source)
        dest = tmp_path / "shallow"
        # A real --depth 1 clone over file:// reproduces a benchmark shallow clone.
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", source.as_uri(), str(dest)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        priors = load_churn_priors(dest)
        assert priors.source == CHURN_SOURCE_NEUTRAL
        assert priors.priors == {}
        assert priors.prior_for("hot.py") == 1.0
