"""Measure the token headroom a repository-context tool has on SWE-bench Pro.

SWE-bench Pro reports resolve rate, not tokens. The question this answers is the
one that decides whether an archex-vs-no-archex token study on that benchmark is
worth funding: **how much of an agent's billed input token spend could a
retrieval tool physically displace, and what does the tool's own bundle cost?**

Inputs are Scale's published SWE-agent trajectories, which live in a publicly
readable S3 bucket (no credentials): ``s3://scaleapi-results/swe-bench-pro/``.
Gold patch file sets come from the ``ScaleAI/SWE-bench_Pro`` HuggingFace dataset.
Nothing here runs a model, a container, or archex; it is arithmetic over evidence
someone else already paid for.

Two stages, so the expensive one runs once:

```bash
# ~13 GB streamed per run, nothing retained on disk
uv run python scripts/swebench_pro_token_headroom.py collect \
    --run claude-45sonnet-10132025 \
    --output benchmarks/swebench_pro/records-claude-45sonnet-10132025.json

uv run python scripts/swebench_pro_token_headroom.py analyze \
    --records benchmarks/swebench_pro/records-claude-45sonnet-10132025.json \
    --records benchmarks/swebench_pro/records-gpt-5-250-turns-10132025.json \
    --output benchmarks/evidence/swebench-pro-token-headroom.json
```

Definitions, pinned before the corpus was read past the 12-instance pilot:

* **Token estimator.** SWE-agent trajectories record the exact context sent to
  the model (``query``) but not its token count. Characters are converted at
  4 chars/token, then rescaled per instance by that instance's own measured bias
  ``sum(len(query)/4) / tokens_sent``, so every reported quantity is a fraction
  of the provider-billed ``tokens_sent`` and the estimator's scale cancels.
* **Compounding.** SWE-agent resends the whole history every call. An
  observation produced at step ``i`` of an ``n``-step trajectory appears in the
  contexts of steps ``i+1 … n-1``, so it costs ``T * (n - 1 - i)`` billed input
  tokens. Every share below is compounded unless labelled ``once``.
* **Channels.** Each step is attributed to exactly one channel by its action:
  ``search`` (grep/rg/find/ls/tree/git-grep), ``read_file`` (editor view, cat,
  head, tail, sed -n, nl), ``edit`` (editor create/str_replace/insert/undo),
  ``test_build`` (test or build runners), ``submit_vcs`` (submit, git diff/status),
  ``other``.
* **Displaceable.** ``search`` observations in full, plus ``read_file``
  observations of files absent from both the gold patch and the test patch. Reads
  of files the accepted solution touched are not displaceable: the agent must see
  their exact lines to patch them. This is an upper bound — some out-of-patch
  reads are genuinely load-bearing, and a retrieval tool would have to supply
  that content at its own token cost.
* **Break-even bundle.** A retrieval tool front-loads context, so its bundle
  takes the largest multiplier. Delivered at step 1 it is charged in steps
  ``2 … n-1``, i.e. ``n - 2`` times. Break-even bundle size is therefore
  ``displaceable / (n - 2)``.
* **Inference.** Repositories are the clusters; 10 000-resample percentile
  cluster bootstrap, seed 20260909, matching every other interval in this repo.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUCKET = "https://scaleapi-results.s3.amazonaws.com"
PREFIX = "swe-bench-pro"
EVAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/traj/{run}/eval_results.json"
)
DATASET_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=ScaleAI%2FSWE-bench_Pro&config=default&split=test&offset={offset}&length=100"
)
DATASET_ROWS = 731

CHARS_PER_TOKEN = 4.0
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260909
BUNDLE_SIZES = (2_500, 5_000, 7_500, 10_000, 13_247, 20_000)
#: archex's own checked-in external-localization bundle size, from
#: benchmarks/cross-tool-efficiency/cross-tool-comparison.json.
ARCHEX_REFERENCE_BUNDLE = 13_247

CHANNELS = ("search", "read_file", "edit", "test_build", "submit_vcs", "other")

_SEARCH = re.compile(
    r"^\s*(?:grep|rg|egrep|fgrep|find|ls|tree|git\s+grep|git\s+ls-files|locate|which)\b"
)
_VIEW = re.compile(
    r"^\s*(?:str_replace_editor\s+view|cat|nl|head|tail|less|more|sed\s+-n\s+\S+)\s+(\S+)"
)
_VIEW_ANY = re.compile(r"^\s*(?:str_replace_editor\s+view|cat|nl|head|tail|less|more|sed\s+-n)\b")
_EDIT = re.compile(
    r"^\s*(?:str_replace_editor\s+(?:create|str_replace|insert|undo_edit)|patch|apply_patch)\b"
)
_TEST = re.compile(
    r"\b(?:pytest|go\s+test|gotestsum|ginkgo|npm\s+(?:test|run)|yarn\s+(?:test|run)|pnpm\s+run"
    r"|mocha|jest|vitest|tox|nosetests|python\s+-m\s+unittest|make\s+\S*test|make\s+build"
    r"|cargo\s+test|mvn\s+test|gradle\s+test)\b"
)
_VCS = re.compile(r"^\s*(?:submit|git\s+(?:diff|status|log|stash|checkout|add|apply|reset))\b")
_DIFF_FILE = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)

_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _get(url: str, retries: int = 5, max_time_s: int = 150) -> bytes:
    """Fetch one object through ``curl``, which enforces its own transfer timeout.

    These objects are 15-25 MB each and S3 occasionally hangs a parallel GET
    without closing the socket. Python's socket timeout does not reliably escape
    that state, so the fetch runs as a subprocess with ``--max-time``: a hung
    transfer becomes a non-zero exit and a retry instead of a wedged worker.
    """
    last = ""
    for attempt in range(retries):
        completed = subprocess.run(  # noqa: S603 - fixed argv, URL is not shell-interpreted
            [
                "curl",
                "-fsSL",
                "--max-time",
                str(max_time_s),
                "--speed-limit",
                "10000",
                "--speed-time",
                "20",
                url,
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout:
            return completed.stdout
        last = f"exit {completed.returncode}: {completed.stderr.decode()[:200]}"
        time.sleep(min(1.5 * (attempt + 1), 8.0))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def channel_of(action: str) -> str:
    """Attribute one trajectory step to exactly one token channel."""
    text = (action or "").strip()
    if not text:
        return "other"
    if _TEST.search(text):
        return "test_build"
    if _EDIT.match(text):
        return "edit"
    if _SEARCH.match(text) or "| grep" in text or "grep -" in text:
        return "search"
    if _VIEW_ANY.match(text):
        return "read_file"
    if _VCS.match(text):
        return "submit_vcs"
    return "other"


def _viewed_path(action: str) -> str | None:
    match = _VIEW.match((action or "").strip())
    if match is None:
        return None
    path = match.group(1).strip("'\"")
    for prefix in ("/app/", "app/", "./"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    return path


def _touches(path: str, targets: frozenset[str]) -> bool:
    return any(
        path == target or path.endswith("/" + target) or target.endswith("/" + path)
        for target in targets
    )


@dataclass(frozen=True)
class Instance:
    """The dataset facts one trajectory is scored against."""

    instance_id: str
    repo: str
    gold_files: frozenset[str]


def load_instances() -> dict[str, Instance]:
    """Gold + test patch file sets for all 731 public instances."""
    instances: dict[str, Instance] = {}
    for offset in range(0, DATASET_ROWS, 100):
        payload = json.loads(_get(DATASET_ROWS_URL.format(offset=offset)))
        for entry in payload["rows"]:
            row = entry["row"]
            files = frozenset(
                _DIFF_FILE.findall(row["patch"] or "") + _DIFF_FILE.findall(row["test_patch"] or "")
            )
            instances[row["instance_id"]] = Instance(
                instance_id=row["instance_id"], repo=row["repo"], gold_files=files
            )
    if len(instances) != DATASET_ROWS:
        raise RuntimeError(f"expected {DATASET_ROWS} dataset rows, got {len(instances)}")
    return instances


def list_trajectories(run: str) -> list[str]:
    """Every ``.traj`` object key for one published run."""
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={PREFIX}/{run}/traj/&max-keys=1000"
        if token is not None:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        tree = ET.fromstring(_get(url))
        for contents in tree.findall("s:Contents", _S3_NS):
            key = contents.findtext("s:Key", namespaces=_S3_NS) or ""
            if key.endswith(".traj"):
                keys.append(key)
        nxt = tree.findtext("s:NextContinuationToken", namespaces=_S3_NS)
        if not nxt:
            return sorted(keys)
        token = nxt


def measure(key: str, instances: dict[str, Instance], resolved: dict[str, bool]) -> dict[str, Any]:
    """Stream one trajectory and reduce it to a per-instance record."""
    name = key.rsplit("/", 1)[-1]
    instance_id = name[: -len(".traj")]
    instance = instances.get(instance_id)
    if instance is None:
        raise RuntimeError(f"no dataset row for {instance_id}")
    payload = json.loads(_get(f"{BUCKET}/{urllib.parse.quote(key)}"))
    steps = payload["trajectory"]
    stats = payload["info"]["model_stats"]
    n = len(steps)
    if n < 3:
        raise RuntimeError(f"{instance_id}: {n} steps, cannot compound")

    contexts = [len(str(step.get("query", ""))) / CHARS_PER_TOKEN for step in steps]
    billed = int(stats["tokens_sent"])
    bias = sum(contexts) / billed if billed else 1.0

    once: dict[str, float] = dict.fromkeys(CHANNELS, 0.0)
    comp: dict[str, float] = dict.fromkeys(CHANNELS, 0.0)
    calls: Counter[str] = Counter()
    displaceable = 0.0
    in_patch_read = 0.0
    for index, step in enumerate(steps):
        action = step.get("action") or ""
        name_ = channel_of(action)
        observed = len(str(step.get("observation", ""))) / CHARS_PER_TOKEN / bias
        weight = observed * (n - 1 - index)
        once[name_] += observed
        comp[name_] += weight
        calls[name_] += 1
        if name_ == "search":
            displaceable += weight
        elif name_ == "read_file":
            path = _viewed_path(action)
            if path is not None and _touches(path, instance.gold_files):
                in_patch_read += weight
            else:
                displaceable += weight

    return {
        "instance_id": instance_id,
        "repo": instance.repo,
        "resolved": resolved.get(instance_id),
        "turns": n,
        "billed_input_tokens": billed,
        "output_tokens": int(stats["tokens_received"]),
        "api_calls": int(stats["api_calls"]),
        "reported_cost_usd": float(stats["instance_cost"]),
        "exit_status": payload["info"].get("exit_status"),
        "estimator_bias": bias,
        "context_first": contexts[0] / bias,
        "context_last": contexts[-1] / bias,
        "context_max": max(contexts) / bias,
        "gold_files": len(instance.gold_files),
        "calls_by_channel": dict(calls),
        "observation_tokens_once": once,
        "observation_tokens_compounded": comp,
        "displaceable_tokens": displaceable,
        "in_patch_read_tokens": in_patch_read,
        "breakeven_bundle_tokens": displaceable / (n - 2),
    }


def _existing_records(output: Path) -> dict[str, dict[str, Any]]:
    """Records already measured into ``output``, so an interrupted run resumes."""
    if not output.exists():
        return {}
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {record["instance_id"]: record for record in payload["records"]}


def _sample_per_repo(
    keys: list[str], instances: dict[str, Instance], limit: int | None
) -> list[str]:
    """Deterministic first-N-per-repository sample, ordered round-robin.

    Membership is the first ``limit`` instance ids of each repository, so the
    population is reproducible. The *order* interleaves repositories, so an
    interrupted run leaves balanced repository coverage rather than a prefix of
    the alphabet — which matters because inference clusters on repository.
    """
    by_repo: dict[str, list[str]] = {}
    for key in keys:
        instance = instances.get(key.rsplit("/", 1)[-1][: -len(".traj")])
        if instance is None:
            continue
        bucket = by_repo.setdefault(instance.repo, [])
        if limit is None or len(bucket) < limit:
            bucket.append(key)
    ordered: list[str] = []
    for depth in range(max((len(bucket) for bucket in by_repo.values()), default=0)):
        for repo in sorted(by_repo):
            if depth < len(by_repo[repo]):
                ordered.append(by_repo[repo][depth])
    return ordered


def collect(run: str, output: Path, workers: int, limit_per_repo: int | None) -> None:
    instances = load_instances()
    resolved: dict[str, bool] = json.loads(_get(EVAL_RESULTS_URL.format(run=run)))
    keys = _sample_per_repo(list_trajectories(run), instances, limit_per_repo)
    done_records = _existing_records(output)
    pending = [key for key in keys if key.rsplit("/", 1)[-1][: -len(".traj")] not in done_records]
    print(
        f"{run}: {len(keys)} selected, {len(done_records)} already measured, "
        f"{len(pending)} to fetch",
        file=sys.stderr,
    )

    def _one(key: str) -> dict[str, Any]:
        return measure(key, instances, resolved)

    selected = {key.rsplit("/", 1)[-1][: -len(".traj")] for key in keys}
    records = [record for record in done_records.values() if record["instance_id"] in selected]

    def _flush() -> None:
        records.sort(key=lambda record: record["instance_id"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "run": run,
                    "population": (
                        "all published trajectories"
                        if limit_per_repo is None
                        else f"first {limit_per_repo} per repository by instance id"
                    ),
                    "instances": len(records),
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # Completion order, not submission order: one slow object must not gate the
    # progress counter or the checkpoint behind it.
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, key) for key in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if done % 10 == 0 or done == len(pending):
                    print(f"  {done}/{len(pending)}", file=sys.stderr, flush=True)
                    _flush()
    finally:
        _flush()
    print(f"wrote {output} ({len(records)} records)", file=sys.stderr)


@dataclass
class Clusters:
    """Per-repository values for the cluster bootstrap."""

    values: dict[str, list[float]] = field(default_factory=lambda: {})

    def add(self, repo: str, value: float) -> None:
        self.values.setdefault(repo, []).append(value)


def _cluster_bootstrap(clusters: Clusters) -> tuple[float, float, float]:
    """Pooled mean and a 95% percentile interval resampling repositories."""
    rng = random.Random(BOOTSTRAP_SEED)
    repos = sorted(clusters.values)
    flat = [value for repo in repos for value in clusters.values[repo]]
    observed = statistics.fmean(flat)
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled: list[float] = []
        for _ in repos:
            sampled.extend(clusters.values[rng.choice(repos)])
        means.append(statistics.fmean(sampled))
    means.sort()
    return (
        observed,
        means[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))],
        means[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))],
    )


def _run_summary(run: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    billed = sum(r["billed_input_tokens"] for r in records)
    channels: dict[str, Any] = {}
    for channel in CHANNELS:
        comp = sum(r["observation_tokens_compounded"][channel] for r in records)
        once = sum(r["observation_tokens_once"][channel] for r in records)
        channels[channel] = {
            "calls": sum(r["calls_by_channel"].get(channel, 0) for r in records),
            "observation_tokens_once": once,
            "observation_tokens_compounded": comp,
            "share_of_billed_input": comp / billed,
        }

    share = Clusters()
    breakeven = Clusters()
    for record in records:
        share.add(record["repo"], record["displaceable_tokens"] / record["billed_input_tokens"])
        breakeven.add(record["repo"], record["breakeven_bundle_tokens"])
    share_mean, share_low, share_high = _cluster_bootstrap(share)
    be_mean, be_low, be_high = _cluster_bootstrap(breakeven)

    curve: list[dict[str, Any]] = []
    for size in BUNDLE_SIZES:
        net = Clusters()
        for record in records:
            cost = size * (record["turns"] - 2)
            net.add(
                record["repo"],
                (record["displaceable_tokens"] - cost) / record["billed_input_tokens"],
            )
        mean, low, high = _cluster_bootstrap(net)
        curve.append(
            {
                "bundle_tokens": size,
                "net_share_of_billed_input": mean,
                "ci95_low": low,
                "ci95_high": high,
                "net_is_a_saving": low > 0.0,
            }
        )

    per_repo = {}
    for repo in sorted(share.values):
        rows = [r for r in records if r["repo"] == repo]
        per_repo[repo] = {
            "instances": len(rows),
            "mean_turns": statistics.fmean(r["turns"] for r in rows),
            "mean_billed_input_tokens": statistics.fmean(r["billed_input_tokens"] for r in rows),
            "mean_reported_cost_usd": statistics.fmean(r["reported_cost_usd"] for r in rows),
            "mean_displaceable_share": statistics.fmean(share.values[repo]),
            "mean_breakeven_bundle_tokens": statistics.fmean(breakeven.values[repo]),
            "resolved_rate": (
                statistics.fmean(1.0 if r["resolved"] else 0.0 for r in rows)
                if all(r["resolved"] is not None for r in rows)
                else None
            ),
        }

    resolved_rows = [r for r in records if r["resolved"] is not None]
    by_outcome = {}
    for label, wanted in (("resolved", True), ("unresolved", False)):
        rows = [r for r in resolved_rows if r["resolved"] is wanted]
        if not rows:
            continue
        by_outcome[label] = {
            "instances": len(rows),
            "mean_turns": statistics.fmean(r["turns"] for r in rows),
            "mean_billed_input_tokens": statistics.fmean(r["billed_input_tokens"] for r in rows),
            "mean_reported_cost_usd": statistics.fmean(r["reported_cost_usd"] for r in rows),
            "mean_displaceable_share": statistics.fmean(
                r["displaceable_tokens"] / r["billed_input_tokens"] for r in rows
            ),
        }

    return {
        "run": run,
        "instances": len(records),
        "repositories": len(share.values),
        "totals": {
            "billed_input_tokens": billed,
            "output_tokens": sum(r["output_tokens"] for r in records),
            "reported_cost_usd": sum(r["reported_cost_usd"] for r in records),
        },
        "per_instance_means": {
            "turns": statistics.fmean(r["turns"] for r in records),
            "billed_input_tokens": statistics.fmean(r["billed_input_tokens"] for r in records),
            "reported_cost_usd": statistics.fmean(r["reported_cost_usd"] for r in records),
            "context_first": statistics.fmean(r["context_first"] for r in records),
            "context_last": statistics.fmean(r["context_last"] for r in records),
            "estimator_bias": statistics.fmean(r["estimator_bias"] for r in records),
        },
        "channels": channels,
        "displaceable_share_of_billed_input": {
            "mean": share_mean,
            "ci95_low": share_low,
            "ci95_high": share_high,
        },
        "in_patch_read_share_of_billed_input": sum(r["in_patch_read_tokens"] for r in records)
        / billed,
        "breakeven_bundle_tokens": {"mean": be_mean, "ci95_low": be_low, "ci95_high": be_high},
        "archex_reference_bundle_tokens": ARCHEX_REFERENCE_BUNDLE,
        "bundle_net_effect_curve": curve,
        "by_outcome": by_outcome,
        "per_repository": per_repo,
    }


def analyze(record_paths: list[Path], output: Path) -> None:
    runs: list[dict[str, Any]] = []
    for path in record_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = _run_summary(payload["run"], payload["records"])
        summary["population"] = payload.get("population", "all published trajectories")
        runs.append(summary)

    evidence = {
        "analysis": "swebench-pro-token-headroom",
        "question": (
            "What share of an agent's billed input tokens on SWE-bench Pro could a "
            "repository-context tool displace, and what may its own bundle cost?"
        ),
        "sources": {
            "trajectories": f"s3://scaleapi-results/{PREFIX}/<run>/traj/ (public, no credentials)",
            "eval_results": EVAL_RESULTS_URL.format(run="<run>"),
            "dataset": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
        },
        "method": {
            "chars_per_token": CHARS_PER_TOKEN,
            "estimator": "per-instance rescale to provider tokens_sent",
            "compounding": "observation at step i charged in steps i+1..n-1",
            "bundle_charged_turns": "n-2 (front-loaded at step 1)",
            "displaceable": "all search observations + reads of files outside gold+test patch",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "clusters": "source repository",
        },
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    for summary in runs:
        share = summary["displaceable_share_of_billed_input"]
        be = summary["breakeven_bundle_tokens"]
        print(
            f"{summary['run']}: {summary['instances']} instances / "
            f"{summary['repositories']} repos | "
            f"search {summary['channels']['search']['share_of_billed_input']:.1%} | "
            f"read_file {summary['channels']['read_file']['share_of_billed_input']:.1%} | "
            f"displaceable {share['mean']:.1%} "
            f"[{share['ci95_low']:.1%}, {share['ci95_high']:.1%}] | "
            f"break-even bundle {be['mean']:,.0f} tok "
            f"[{be['ci95_low']:,.0f}, {be['ci95_high']:,.0f}]"
        )
    print(f"wrote {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="stream one published run into records")
    collect_parser.add_argument("--run", required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--workers", type=int, default=12)
    collect_parser.add_argument(
        "--limit-per-repo",
        type=int,
        default=None,
        help="measure at most N instances per repository (deterministic by instance id)",
    )

    analyze_parser = sub.add_parser("analyze", help="derive the evidence artifact from records")
    analyze_parser.add_argument("--records", type=Path, action="append", required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.run, args.output, args.workers, args.limit_per_repo)
    else:
        analyze(args.records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
