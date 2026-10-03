from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from archex.benchmark.swe_ab import SweAbArm, SweAbError, load_campaign, load_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
MUNA = load_campaign("benchmarks/swe_ab/campaigns/muna.yml", root=REPO_ROOT)
FREE = load_campaign("benchmarks/swe_ab/campaigns/openrouter-space-bunny.yml", root=REPO_ROOT)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

sample: Any = importlib.import_module("swe_ab_sample")

SIZES: dict[str, int] = {
    "org-a/big": 60,
    "org-b/second": 50,
    "org-c/third": 40,
    "element-hq/element-web": 30,
    "org-d/fifth": 25,
    "org-e/sixth": 20,
    "org-f/seventh": 15,
    "org-g/eighth": 12,
    "org-h/ninth": 10,
    "org-i/tenth": 8,
    "org-j/tiny": 4,
}


def _iid(repo: str, n: int) -> str:
    org, name = repo.split("/")
    # real V2 ids carry the -v<suffix> only for some instances
    suffix = f"-v{n}" if n % 3 else ""
    return f"instance_{org}__{name}-{n:040x}{suffix}"


def _make_root(tmp_path: Path, sizes: dict[str, int] = SIZES) -> Path:
    root = tmp_path / "data" / "tasks"
    for repo, count in sizes.items():
        for n in range(count):
            (root / _iid(repo, n)).mkdir(parents=True)
    return root


def _run(
    tmp_path: Path,
    root: Path,
    stage: int,
    *extra: str,
    tag: str = "o",
    campaign: str = MUNA.path,
) -> tuple[int, Path, Path, Path]:
    out = tmp_path / tag
    out.mkdir(exist_ok=True)
    plan, manifest, inst = out / "plan.json", out / "m.json", out / "i.txt"
    code = sample.main(
        [
            "--tasks-root", str(root),
            "--stage", str(stage),
            "--campaign", campaign,
            "--cost-ceiling", "5",
            "--plan-out", str(plan),
            "--manifest-out", str(manifest),
            "--instances-out", str(inst),
            *extra,
        ]
    )  # fmt: skip
    return code, plan, manifest, inst


def _excl(tmp_path: Path, ids: list[str], reason: str = "gold_not_resolved") -> Path:
    path = tmp_path / "excl.json"
    path.write_text(
        json.dumps(
            {"exclusions": [{"instance_id": i, "reason": reason, "source": "t"} for i in ids]}
        )
    )
    return path


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_stage1_allocation_extra_to_two_largest(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    code, plan_path, manifest, inst = _run(tmp_path, root, 1)
    assert code == 0
    alloc = _manifest(manifest)["allocation_by_repo"]
    expected = dict.fromkeys(SIZES, 2)
    expected["org-a/big"] = 3
    expected["org-b/second"] = 3
    assert alloc == expected
    plan = load_plan(plan_path)
    assert len(plan.tasks) == 24
    assert plan.name == "r3x-stage1"
    assert plan.models == list(MUNA.labels)
    assert plan.repetitions == {SweAbArm.A0: 2, SweAbArm.H: 1, SweAbArm.HC: 1, SweAbArm.C: 1}
    assert all(t.image is None and t.task_dir is None for t in plan.tasks)
    assert inst.read_text().split() == [t.task_id for t in plan.tasks]
    repos = [t.repo for t in plan.tasks]
    assert repos == sorted(repos)


def test_stage1_extra_tie_broken_by_repo_name(tmp_path: Path) -> None:
    sizes = {"z/last": 10, "b/beta": 10, "a/alpha": 10, "c/small": 3}
    root = _make_root(tmp_path, sizes)
    code, _, manifest, _ = _run(tmp_path, root, 1)
    assert code == 0
    assert _manifest(manifest)["allocation_by_repo"] == {
        "a/alpha": 3,
        "b/beta": 3,
        "c/small": 2,
        "z/last": 2,
    }


def test_same_inputs_byte_identical(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, p1, m1, i1 = _run(tmp_path, root, 1, tag="a")
    _, p2, m2, i2 = _run(tmp_path, root, 1, tag="b")
    assert p1.read_bytes() == p2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()
    assert i1.read_bytes() == i2.read_bytes()


def test_excluded_replaced_by_next_ranked_in_same_repo(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, _, m0, _ = _run(tmp_path, root, 1, tag="a")
    base = _manifest(m0)
    repo = "element-hq/element-web"
    walk = base["walk_by_repo"][repo]
    victim = walk[0]["instance_id"]
    code, _, m1, _ = _run(
        tmp_path, root, 1, "--exclusions", str(_excl(tmp_path, [victim])), tag="b"
    )
    assert code == 0
    new = _manifest(m1)
    rows = new["walk_by_repo"][repo]
    assert rows[0] == {
        "instance_id": victim,
        "rank": 0,
        "disposition": "excluded:gold_not_resolved",
    }
    ranked = sample.list_pool(root)[repo]
    assert [r["instance_id"] for r in rows] == ranked[: len(rows)]
    picked = [r["instance_id"] for r in rows if r["disposition"] == "selected"]
    assert picked == ranked[1:3]
    assert victim not in new["selected"]
    assert new["allocation_by_repo"] == base["allocation_by_repo"]
    # rank order does not depend on the pool composition
    assert sample.rank_key(ranked[0]) < sample.rank_key(ranked[1])


def test_exhausted_repo_shortfall_goes_to_repo_with_most_remaining(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    tiny = sample.list_pool(root)["org-j/tiny"]
    excl = _excl(tmp_path, tiny[:3], "empty_not_failing")
    code, _, manifest, _ = _run(tmp_path, root, 1, "--exclusions", str(excl))
    assert code == 0
    alloc = _manifest(manifest)["allocation_by_repo"]
    assert alloc["org-j/tiny"] == 1
    assert alloc["org-a/big"] == 4
    assert sum(alloc.values()) == 24


def test_stage2_disjoint_and_hamilton(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, p1, _, _ = _run(tmp_path, root, 1, tag="s1")
    code, p2, m2, _ = _run(
        tmp_path, root, 2, "--stage1-plan", str(p1), "--stage2-tasks", "124", tag="s2"
    )
    assert code == 0
    s1 = {t.task_id for t in load_plan(p1).tasks}
    plan2 = load_plan(p2)
    s2 = [t.task_id for t in plan2.tasks]
    assert len(s2) == 124 and not s1 & set(s2)
    assert plan2.name == "r3x-stage2"
    assert plan2.repetitions == {SweAbArm.A0: 1, SweAbArm.H: 1, SweAbArm.HC: 1}
    manifest = _manifest(m2)
    assert manifest["stage1_plan_task_ids"] == [t.task_id for t in load_plan(p1).tasks]
    assert manifest["stage2_tasks"] == 124
    total = sum(SIZES.values())
    for repo, k in manifest["allocation_by_repo"].items():
        exact = 124 * SIZES[repo] / total
        assert int(exact) <= k <= int(exact) + 1
    assert sum(manifest["allocation_by_repo"].values()) == 124


def test_stage2_shortfall_reassigned(tmp_path: Path) -> None:
    sizes = {"a/one": 30, "b/two": 30, "c/three": 30, "d/four": 30, "e/tiny": 4}
    root = _make_root(tmp_path, sizes)
    _, p1, _, _ = _run(tmp_path, root, 1, tag="s1")
    code, _, m2, _ = _run(
        tmp_path, root, 2, "--stage1-plan", str(p1), "--stage2-tasks", "100", tag="s2"
    )
    assert code == 0
    alloc = _manifest(m2)["allocation_by_repo"]
    assert alloc["e/tiny"] == 2  # 4 in pool, 2 taken by stage 1
    assert sum(alloc.values()) == 100


def test_stage2_without_hc(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, p1, _, _ = _run(tmp_path, root, 1, tag="s1")
    code, p2, _, _ = _run(
        tmp_path,
        root,
        2,
        "--stage1-plan", str(p1), "--stage2-tasks", "100", "--without-hc",
        tag="s2",
    )  # fmt: skip
    assert code == 0
    assert load_plan(p2).repetitions == {SweAbArm.A0: 1, SweAbArm.H: 1}


def test_stage2_refuses_mismatched_stage1_plan(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, p1, m1, _ = _run(tmp_path, root, 1, tag="s1")
    victim = _manifest(m1)["selected"][0]
    excl = _excl(tmp_path, [victim])
    code, p2, _, _ = _run(
        tmp_path,
        root,
        2,
        "--stage1-plan", str(p1), "--stage2-tasks", "100", "--exclusions", str(excl),
        tag="s2",
    )  # fmt: skip
    assert code == 1
    assert not p2.exists()


def test_refusals(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _make_root(tmp_path)
    some = sample.list_pool(root)["org-a/big"][0]
    bad_reason = _excl(tmp_path, [some], "flaky")
    assert _run(tmp_path, root, 1, "--exclusions", str(bad_reason), tag="a")[0] == 1
    unknown = _excl(tmp_path, ["instance_x__y-" + "0" * 40 + "-v1"])
    assert _run(tmp_path, root, 1, "--exclusions", str(unknown), tag="b")[0] == 1
    dup = _excl(tmp_path, [some, some])
    assert _run(tmp_path, root, 1, "--exclusions", str(dup), tag="c")[0] == 1
    (root / "not-an-instance").mkdir()
    assert _run(tmp_path, root, 1, tag="d")[0] == 1
    assert capsys.readouterr().err.count("REFUSED:") == 4


def test_overwrite_requires_force(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    assert _run(tmp_path, root, 1)[0] == 0
    assert _run(tmp_path, root, 1)[0] == 1
    assert _run(tmp_path, root, 1, "--force")[0] == 0


def test_stage_argument_combinations_refused(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    assert _run(tmp_path, root, 2, tag="a")[0] == 1
    assert _run(tmp_path, root, 1, "--without-hc", tag="b")[0] == 1
    assert _run(tmp_path, root, 1, "--stage2-tasks", "100", tag="c")[0] == 1
    assert _run(tmp_path, root, 2, "--stage1-plan", str(tmp_path / "x"), tag="d")[0] == 1


def test_stage2_tasks_below_the_floor_are_refused(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, p1, _, _ = _run(tmp_path, root, 1, tag="s1")
    code, p2, _, _ = _run(
        tmp_path, root, 2, "--stage1-plan", str(p1), "--stage2-tasks", "99", tag="s2"
    )
    assert code == 1
    assert not p2.exists()


@pytest.mark.parametrize(("passing", "tasks"), [(1, 370), (2, 185), (3, 124)])
def test_stage2_task_count_follows_the_size_rule(passing: int, tasks: int) -> None:
    assert sample.stage2_task_count(passing, 3) == tasks


@pytest.mark.parametrize(("passing", "configurations"), [(0, 3), (4, 3), (2, 1)])
def test_stage2_task_count_refuses_counts_outside_the_configurations(
    passing: int, configurations: int
) -> None:
    with pytest.raises(SweAbError):
        sample.stage2_task_count(passing, configurations)


def test_plan_names_the_campaign_and_the_manifest_records_its_hash(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    code, plan_path, manifest, _ = _run(tmp_path, root, 1)
    assert code == 0
    plan = load_plan(plan_path)
    assert plan.campaign == MUNA.path
    assert plan.token_ceiling is None
    assert _manifest(manifest)["campaign"] == MUNA.path
    assert _manifest(manifest)["campaign_sha256"] == MUNA.sha256


def test_a_campaign_is_required(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    with pytest.raises(SystemExit):
        sample.main(
            [
                "--tasks-root", str(root), "--stage", "1", "--cost-ceiling", "5",
                "--plan-out", str(tmp_path / "p.json"),
                "--manifest-out", str(tmp_path / "m.json"),
                "--instances-out", str(tmp_path / "i.txt"),
            ]
        )  # fmt: skip


def test_an_unpriced_campaign_needs_a_token_ceiling_and_writes_nothing_without_one(
    tmp_path: Path,
) -> None:
    root = _make_root(tmp_path)
    code, plan, manifest, inst = _run(tmp_path, root, 1, campaign=FREE.path, tag="no")
    assert code == 1
    assert not (plan.exists() or manifest.exists() or inst.exists())

    code, plan, _, _ = _run(
        tmp_path, root, 1, "--token-ceiling", "5000000", campaign=FREE.path, tag="yes"
    )
    assert code == 0
    written = load_plan(plan)
    assert (written.campaign, written.token_ceiling) == (FREE.path, 5_000_000)
    assert written.models == list(FREE.labels)


def test_a_non_positive_token_ceiling_is_refused(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    assert _run(tmp_path, root, 1, "--token-ceiling", "0", tag="z")[0] == 1


def test_stage2_keeps_the_stage1_campaign_and_refuses_another(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _, p1, _, _ = _run(
        tmp_path, root, 1, "--token-ceiling", "5000000", campaign=FREE.path, tag="s1"
    )
    stage2 = ("--stage1-plan", str(p1), "--stage2-tasks", "100", "--token-ceiling", "5000000")

    code, other, _, _ = _run(tmp_path, root, 2, *stage2, campaign=MUNA.path, tag="bad")
    assert code == 1
    assert not other.exists()

    code, p2, _, _ = _run(tmp_path, root, 2, *stage2, campaign=FREE.path, tag="ok")
    assert code == 0
    assert load_plan(p2).campaign == FREE.path
