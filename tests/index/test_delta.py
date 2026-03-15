"""Tests for delta indexing: compute_delta, apply_delta, compute_mtime_delta."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from archex.exceptions import DeltaIndexError
from archex.index.delta import apply_delta, compute_delta, compute_mtime_delta
from archex.index.store import IndexStore
from archex.models import ChangeStatus, CodeChunk, Config, IndexConfig
from archex.pipeline.service import build_chunk_surrogates

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _git(repo: Path, *args: str) -> str:
    """Run a git command in repo and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def delta_test_repo(tmp_path: Path) -> Path:
    """Create a git repo with initial commit from python_simple fixture."""
    src = FIXTURES_DIR / "python_simple"
    dest = tmp_path / "delta_repo"
    shutil.copytree(src, dest)
    _git(dest, "init")
    _git(dest, "config", "user.email", "test@archex.test")
    _git(dest, "config", "user.name", "archex-test")
    _git(dest, "add", ".")
    _git(dest, "commit", "-m", "initial")
    return dest


# ---------------------------------------------------------------------------
# TestComputeDelta
# ---------------------------------------------------------------------------


class TestComputeDelta:
    def test_modified_file(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        (delta_test_repo / "utils.py").write_text("def new_util(): return 1\n")
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "modify utils")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert manifest.modified_files == ["utils.py"]
        assert manifest.added_files == []
        assert manifest.deleted_files == []

    def test_added_file(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        (delta_test_repo / "new_module.py").write_text("def new_func(): pass\n")
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "add new module")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert manifest.added_files == ["new_module.py"]

    def test_deleted_file(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        (delta_test_repo / "utils.py").unlink()
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "delete utils")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert manifest.deleted_files == ["utils.py"]

    def test_renamed_file(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        _git(delta_test_repo, "mv", "utils.py", "helpers.py")
        _git(delta_test_repo, "commit", "-m", "rename utils to helpers")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert len(manifest.renamed_files) == 1
        old, new = manifest.renamed_files[0]
        assert old == "utils.py"
        assert new == "helpers.py"

    def test_multiple_changes(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        (delta_test_repo / "utils.py").write_text("# modified\n")
        (delta_test_repo / "extra.py").write_text("def extra(): pass\n")
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "multiple changes")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert "utils.py" in manifest.modified_files
        assert "extra.py" in manifest.added_files
        assert len(manifest.all_affected_files) >= 2

    def test_no_changes(self, delta_test_repo: Path) -> None:
        commit = _git_head(delta_test_repo)
        manifest = compute_delta(delta_test_repo, commit, commit)
        assert manifest.changes == []

    def test_invalid_base_commit_raises(self, delta_test_repo: Path) -> None:
        current = _git_head(delta_test_repo)
        with pytest.raises(DeltaIndexError, match="not reachable"):
            compute_delta(delta_test_repo, "deadbeef" * 5, current)

    def test_nonexistent_repo_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DeltaIndexError):
            compute_delta(tmp_path / "nonexistent", "abc", "def")

    def test_manifest_commits_recorded(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        (delta_test_repo / "utils.py").write_text("# changed\n")
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "change")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert manifest.base_commit == base
        assert manifest.current_commit == current

    def test_change_status_types(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        (delta_test_repo / "utils.py").write_text("# modified\n")
        (delta_test_repo / "new.py").write_text("x = 1\n")
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "mixed")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        statuses = {c.status for c in manifest.changes}
        assert ChangeStatus.MODIFIED in statuses
        assert ChangeStatus.ADDED in statuses

    def test_all_affected_files_includes_renames(self, delta_test_repo: Path) -> None:
        base = _git_head(delta_test_repo)
        _git(delta_test_repo, "mv", "utils.py", "helpers.py")
        _git(delta_test_repo, "commit", "-m", "rename")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        affected = manifest.all_affected_files
        assert "utils.py" in affected
        assert "helpers.py" in affected


# ---------------------------------------------------------------------------
# TestApplyDelta
# ---------------------------------------------------------------------------


class TestApplyDelta:
    def _build_initial_index(self, repo_path: Path, tmp_path: Path) -> tuple[IndexStore, object]:
        """Build a full index for the repo. Returns (store, graph)."""
        from archex.acquire import discover_files
        from archex.index.bm25 import BM25Index
        from archex.index.chunker import ASTChunker
        from archex.index.graph import DependencyGraph
        from archex.parse import (
            TreeSitterEngine,
            build_file_map,
            extract_symbols,
            parse_imports,
            resolve_imports,
        )
        from archex.parse.adapters import default_adapter_registry

        files = discover_files(repo_path)
        engine = TreeSitterEngine()
        adapters = default_adapter_registry.build_all()
        parsed = extract_symbols(files, engine, adapters)
        import_map = parse_imports(files, engine, adapters)
        file_map = build_file_map(files)
        file_langs = {f.path: f.language for f in files}
        resolved = resolve_imports(import_map, file_map, adapters, file_langs)
        graph = DependencyGraph.from_parsed_files(parsed, resolved)

        chunker = ASTChunker(config=IndexConfig())
        sources: dict[str, bytes] = {}
        for f in files:
            try:
                sources[f.path] = Path(f.absolute_path).read_bytes()
            except OSError:
                continue
        chunks = chunker.chunk_files(parsed, sources)

        db_path = tmp_path / "test_index.db"
        store = IndexStore(db_path)
        store.insert_chunks(chunks)
        store.insert_chunk_surrogates(build_chunk_surrogates(chunks))
        store.insert_edges(graph.file_edges())
        bm25 = BM25Index(store)
        bm25.build(chunks)
        return store, graph

    def test_modified_replaces_chunks(self, delta_test_repo: Path, tmp_path: Path) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            old_chunks = store.get_chunks_for_file("utils.py")
            assert len(old_chunks) > 0

            base = _git_head(delta_test_repo)
            (delta_test_repo / "utils.py").write_text("def brand_new_util():\n    return 42\n")
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "modify utils")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            config = Config(cache=False)
            meta = apply_delta(store, graph, manifest, delta_test_repo, config)

            new_chunks = store.get_chunks_for_file("utils.py")
            assert len(new_chunks) > 0
            all_content = " ".join(c.content for c in new_chunks)
            assert "brand_new_util" in all_content
            assert meta.files_modified == 1
            assert meta.full_reindex_avoided is True
        finally:
            store.close()

    def test_modified_rebuilds_chunk_surrogates(
        self,
        delta_test_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            base = _git_head(delta_test_repo)
            (delta_test_repo / "utils.py").write_text("def brand_new_util():\n    return 42\n")
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "modify utils")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            apply_delta(store, graph, manifest, delta_test_repo, Config(cache=False))

            surrogates = store.get_chunk_surrogates_for_file("utils.py")
            assert surrogates
            assert any("brand_new_util" in surrogate.surrogate_text for surrogate in surrogates)
        finally:
            store.close()

    def test_added_inserts_chunks(self, delta_test_repo: Path, tmp_path: Path) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            base = _git_head(delta_test_repo)
            (delta_test_repo / "brand_new.py").write_text("def hello():\n    return 'world'\n")
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "add file")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            config = Config(cache=False)
            meta = apply_delta(store, graph, manifest, delta_test_repo, config)

            chunks = store.get_chunks_for_file("brand_new.py")
            assert len(chunks) > 0
            assert meta.files_added == 1
        finally:
            store.close()

    def test_deleted_file_removes_chunk_surrogates(
        self,
        delta_test_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            assert store.get_chunk_surrogates_for_file("utils.py")
            base = _git_head(delta_test_repo)
            (delta_test_repo / "utils.py").unlink()
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "delete utils")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            apply_delta(store, graph, manifest, delta_test_repo, Config(cache=False))

            assert store.get_chunk_surrogates_for_file("utils.py") == []
        finally:
            store.close()

    def test_deleted_removes_all(self, delta_test_repo: Path, tmp_path: Path) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            assert len(store.get_chunks_for_file("utils.py")) > 0

            base = _git_head(delta_test_repo)
            (delta_test_repo / "utils.py").unlink()
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "delete utils")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            config = Config(cache=False)
            meta = apply_delta(store, graph, manifest, delta_test_repo, config)

            assert store.get_chunks_for_file("utils.py") == []
            assert meta.files_deleted == 1
        finally:
            store.close()

    def test_metadata_updated(self, delta_test_repo: Path, tmp_path: Path) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            base = _git_head(delta_test_repo)
            (delta_test_repo / "utils.py").write_text("# changed\n")
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "change")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            apply_delta(store, graph, manifest, delta_test_repo, Config(cache=False))

            assert store.get_metadata("commit_hash") == current
            assert store.get_metadata("delta_applied") == "true"
            assert store.get_metadata("file_count") is not None
        finally:
            store.close()

    def test_empty_manifest_no_op(self, delta_test_repo: Path, tmp_path: Path) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            commit = _git_head(delta_test_repo)
            # Same base and current: no changes
            manifest = compute_delta(delta_test_repo, commit, commit)
            chunk_count_before = len(store.get_chunks())
            meta = apply_delta(store, graph, manifest, delta_test_repo, Config(cache=False))
            chunk_count_after = len(store.get_chunks())
            assert chunk_count_before == chunk_count_after
            assert meta.files_modified == 0
            assert meta.files_added == 0
            assert meta.files_deleted == 0
        finally:
            store.close()

    def test_delta_meta_full_reindex_avoided(self, delta_test_repo: Path, tmp_path: Path) -> None:
        from archex.index.graph import DependencyGraph

        store, graph = self._build_initial_index(delta_test_repo, tmp_path)
        assert isinstance(graph, DependencyGraph)
        try:
            base = _git_head(delta_test_repo)
            (delta_test_repo / "utils.py").write_text("# small change\n")
            _git(delta_test_repo, "add", ".")
            _git(delta_test_repo, "commit", "-m", "small")
            current = _git_head(delta_test_repo)

            manifest = compute_delta(delta_test_repo, base, current)
            meta = apply_delta(store, graph, manifest, delta_test_repo, Config(cache=False))
            assert meta.full_reindex_avoided is True
            assert meta.delta_time_ms >= 0
        finally:
            store.close()

    def test_delta_import_resolution_uses_full_file_map(self, tmp_path: Path) -> None:
        """Verify that delta re-parse resolves imports targeting unchanged files.

        File A imports from file B. Only A is modified in the delta. The edge
        A→B must be present after apply_delta, proving build_file_map received
        all_files (including unchanged B) rather than only changed_files.
        """
        from archex.index.graph import DependencyGraph

        repo = tmp_path / "import_repo"
        repo.mkdir()
        (repo / "library.py").write_text("def helper():\n    return 1\n")
        (repo / "consumer.py").write_text(
            "from library import helper\n\ndef run():\n    return helper()\n"
        )
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@archex.test")
        _git(repo, "config", "user.name", "archex-test")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "initial")

        db_dir = tmp_path / "db_initial"
        db_dir.mkdir()
        store, graph = self._build_initial_index(repo, db_dir)
        assert isinstance(graph, DependencyGraph)
        try:
            initial_edges = {(e.source, e.target) for e in store.get_edges()}
            assert ("consumer.py", "library.py") in initial_edges, (
                "initial index must have consumer.py -> library.py edge"
            )

            base = _git_head(repo)
            # Modify only consumer.py; library.py is unchanged
            (repo / "consumer.py").write_text(
                "from library import helper\n\ndef run():\n    return helper() + 1\n"
            )
            _git(repo, "add", ".")
            _git(repo, "commit", "-m", "modify consumer only")
            current = _git_head(repo)

            manifest = compute_delta(repo, base, current)
            assert manifest.modified_files == ["consumer.py"]
            assert "library.py" not in manifest.modified_files

            apply_delta(store, graph, manifest, repo, Config(cache=False))

            after_edges = {(e.source, e.target) for e in store.get_edges()}
            assert ("consumer.py", "library.py") in after_edges, (
                "edge consumer.py -> library.py must survive delta when only consumer.py changed"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# TestComputeMtimeDelta
# ---------------------------------------------------------------------------


class TestComputeMtimeDelta:
    def test_detects_modified(self, tmp_path: Path) -> None:
        """File with mtime > last_indexed is detected as modified."""
        (tmp_path / "mod.py").write_text("x = 1\n")
        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            chunk = CodeChunk(
                id="mod.py::x#variable",
                content="x = 1",
                file_path="mod.py",
                start_line=1,
                end_line=1,
                language="python",
            )
            store.insert_chunks([chunk])

            indexed_at = time.time() - 10  # 10 seconds ago

            # Touch the file to update mtime
            (tmp_path / "mod.py").write_text("x = 2\n")

            manifest = compute_mtime_delta(tmp_path, store, indexed_at)
            assert any(
                c.path == "mod.py" and c.status == ChangeStatus.MODIFIED for c in manifest.changes
            )
        finally:
            store.close()

    def test_detects_added(self, tmp_path: Path) -> None:
        """File on disk but not in store is detected as added."""
        (tmp_path / "new.py").write_text("y = 2\n")
        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            manifest = compute_mtime_delta(tmp_path, store, time.time() - 10)
            assert any(
                c.path == "new.py" and c.status == ChangeStatus.ADDED for c in manifest.changes
            )
        finally:
            store.close()

    def test_detects_deleted(self, tmp_path: Path) -> None:
        """File in store but not on disk is detected as deleted."""
        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            chunk = CodeChunk(
                id="gone.py::f#function",
                content="def f(): pass",
                file_path="gone.py",
                start_line=1,
                end_line=1,
                language="python",
            )
            store.insert_chunks([chunk])
            manifest = compute_mtime_delta(tmp_path, store, time.time() - 10)
            assert any(
                c.path == "gone.py" and c.status == ChangeStatus.DELETED for c in manifest.changes
            )
        finally:
            store.close()

    def test_unchanged_file_not_reported(self, tmp_path: Path) -> None:
        """File older than last_indexed_at is not in the manifest."""
        py_file = tmp_path / "stable.py"
        py_file.write_text("z = 3\n")

        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            chunk = CodeChunk(
                id="stable.py::z#variable",
                content="z = 3",
                file_path="stable.py",
                start_line=1,
                end_line=1,
                language="python",
            )
            store.insert_chunks([chunk])

            # Set last_indexed_at to future so file appears unchanged
            indexed_at = time.time() + 3600

            manifest = compute_mtime_delta(tmp_path, store, indexed_at)
            modified = [c for c in manifest.changes if c.status == ChangeStatus.MODIFIED]
            assert not any(c.path == "stable.py" for c in modified)
        finally:
            store.close()

    def test_empty_store_all_files_added(self, tmp_path: Path) -> None:
        """All discovered files are ADDED when store is empty."""
        (tmp_path / "a.py").write_text("a = 1\n")
        (tmp_path / "b.py").write_text("b = 2\n")
        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            manifest = compute_mtime_delta(tmp_path, store, time.time() - 10)
            added = [c.path for c in manifest.changes if c.status == ChangeStatus.ADDED]
            assert "a.py" in added
            assert "b.py" in added
        finally:
            store.close()

    def test_mtime_manifest_uses_mtime_commits(self, tmp_path: Path) -> None:
        """Mtime-based manifest uses 'mtime' as both base and current commit."""
        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            manifest = compute_mtime_delta(tmp_path, store, time.time())
            assert manifest.base_commit == "mtime"
            assert manifest.current_commit == "mtime"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestComputeDeltaEdgeCases:
    def test_unreachable_current_commit_raises(self, delta_test_repo: Path) -> None:
        """Unreachable current commit raises DeltaIndexError."""
        base = _git_head(delta_test_repo)
        with pytest.raises(DeltaIndexError):
            compute_delta(delta_test_repo, base, "deadbeef" * 5)

    def test_manifest_properties_on_additions_only(self, delta_test_repo: Path) -> None:
        """Manifest with only additions has correct property values."""
        base = _git_head(delta_test_repo)
        (delta_test_repo / "new1.py").write_text("x = 1\n")
        (delta_test_repo / "new2.py").write_text("y = 2\n")
        _git(delta_test_repo, "add", ".")
        _git(delta_test_repo, "commit", "-m", "add two files")
        current = _git_head(delta_test_repo)

        manifest = compute_delta(delta_test_repo, base, current)
        assert len(manifest.added_files) == 2
        assert manifest.modified_files == []
        assert manifest.deleted_files == []
        assert manifest.renamed_files == []


class TestComputeMtimeDeltaEdgeCases:
    def test_mtime_equal_to_last_indexed_not_modified(self, tmp_path: Path) -> None:
        """File with mtime == last_indexed_at is NOT detected as modified (strict >)."""
        py_file = tmp_path / "exact.py"
        py_file.write_text("x = 1\n")
        mtime = py_file.stat().st_mtime

        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            chunk = CodeChunk(
                id="exact.py::x#variable",
                content="x = 1",
                file_path="exact.py",
                start_line=1,
                end_line=1,
                language="python",
            )
            store.insert_chunks([chunk])
            # Set last_indexed_at to exactly the file's mtime
            manifest = compute_mtime_delta(tmp_path, store, mtime)
            modified = [c for c in manifest.changes if c.status == ChangeStatus.MODIFIED]
            assert not any(c.path == "exact.py" for c in modified)
        finally:
            store.close()

    def test_oserror_on_stat_skips_file(self, tmp_path: Path) -> None:
        """File that raises OSError on stat is silently skipped."""
        # Create a file, index it, then remove it to simulate deletion/OSError
        py_file = tmp_path / "perm.py"
        py_file.write_text("z = 3\n")

        db = tmp_path / "test.db"
        store = IndexStore(db)
        try:
            chunk = CodeChunk(
                id="perm.py::z#variable",
                content="z = 3",
                file_path="perm.py",
                start_line=1,
                end_line=1,
                language="python",
            )
            store.insert_chunks([chunk])

            # Remove the file to simulate OSError on stat
            py_file.unlink()

            # Should detect as deleted, not crash
            manifest = compute_mtime_delta(tmp_path, store, time.time() - 10)
            deleted = [c for c in manifest.changes if c.status == ChangeStatus.DELETED]
            assert any(c.path == "perm.py" for c in deleted)
        finally:
            store.close()
