"""Run one SWE A/B cell — (task, model, arm, repetition) — and write its artifact.

Reads a JSON cell spec on stdin (the suite writes it) and writes one
`SweAbCell` JSON to the spec's `output`. Every outcome, including a harness
crash, becomes an artifact: a cell is never silently dropped.

Two runtimes share one flow:

* ``docker`` — the campaign runtime (spec §5). The official SWE-bench Pro V2
  image runs with the pinned omp linux-x64 build and a copy of the provisioned
  `swebench` profile; archex is installed into ``/opt/archex`` with its own
  uv-managed Python; the patch is scored in a fresh container by the task's own
  verifier (``tests/test.sh`` → ``/logs/verifier/reward.txt``).
* ``local`` — the no-spend rehearsal runtime: a copy of a local git
  repository, the host's omp and archex, an isolated ``HOME``, and no scoring.

Flow: prepare → (H/HC/C) exclude ``.archex/``, ``archex init --no-index``,
restore any ``.gitignore`` edit, ``archex index`` → (H/HC) render the hook
module → run omp with the frozen command line → parse the session and the
hook ledger → take ``git diff`` against the base commit (``.archex/`` is
excluded) → score → record.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from archex.benchmark.swe_ab import (
    ANNOTATION_MARKER,
    BASE_TOOLS,
    CHANNELS,
    CLI_GUIDE_PATH,
    COMPRESSOR_MARKERS,
    MAX_TIME,
    OMP_VERSION,
    CellStatus,
    FailureReason,
    HookLedgerSummary,
    Isolation,
    Localization,
    OmpSession,
    SweAbArm,
    SweAbCell,
    Usage,
    archex_subcommand,
    compound,
    diff_files,
    localize,
    observations,
    omp_argv,
    parse_omp_session,
    provider_endpoint_overridden,
    read_ledger,
    sha256_file,
    summarize_ledger,
    system_prompt_violations,
    tool_fingerprint,
)
from archex.client_setup import render_annotation_hook_module
from archex.reporting import count_tokens

_MAX_TIME_SECONDS = int(MAX_TIME.removesuffix("m")) * 60
_OMP_GRACE_SECONDS = 600
_VERIFIER_TIMEOUT_SECONDS = 3000
_SETUP_TIMEOUT_SECONDS = 3600


class CellSpec:
    """The suite's instructions for one cell (see `run_swe_ab_suite.py`)."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.task_id = str(raw["task_id"])
        self.repo = str(raw["repo"])
        self.model = str(raw["model"])
        self.arm = SweAbArm(str(raw["arm"]))
        self.repetition = int(raw["repetition"])
        self.runtime = str(raw["runtime"])
        self.output = Path(str(raw["output"]))
        self.work_dir = Path(str(raw["work_dir"]))
        self.omp_command = [str(part) for part in raw["omp_command"]]
        self.profile_dir = Path(str(raw["profile_dir"]))
        self.cli_guide = Path(str(raw.get("cli_guide") or CLI_GUIDE_PATH))
        self.capture_dir = Path(str(raw["capture_dir"])) if raw.get("capture_dir") else None
        self.image = str(raw.get("image") or "local")
        self.task_dir = Path(str(raw["task_dir"])) if raw.get("task_dir") else None
        self.local_repo = Path(str(raw["local_repo"])) if raw.get("local_repo") else None
        self.prompt_file = Path(str(raw["prompt_file"])) if raw.get("prompt_file") else None
        self.gold_patch = Path(str(raw["gold_patch"])) if raw.get("gold_patch") else None
        self.archex_python = str(raw.get("archex_python") or sys.executable)
        self.archex_wheel = Path(str(raw["archex_wheel"])) if raw.get("archex_wheel") else None
        self.uv_binary = Path(str(raw["uv_binary"])) if raw.get("uv_binary") else None
        self.omp_dir = Path(str(raw["omp_dir"])) if raw.get("omp_dir") else None
        self.network = str(raw.get("network") or "bridge")


class Runtime(Protocol):
    repo: str
    home: str
    out: str

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = _SETUP_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]: ...

    def put(self, source: Path, target: str) -> None: ...

    def get(self, source: str, target: Path) -> None: ...

    def close(self) -> None: ...


def _checked(done: subprocess.CompletedProcess[str], what: str) -> str:
    if done.returncode != 0:
        raise RuntimeError(
            f"{what} failed ({done.returncode}): {(done.stderr or done.stdout)[-800:]}"
        )
    return done.stdout


class LocalRuntime:
    """A private copy of a local git repository, run on the host."""

    def __init__(self, spec: CellSpec) -> None:
        if spec.local_repo is None:
            raise ValueError("the local runtime needs `local_repo`")
        root = spec.work_dir
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        self.repo = str(root / "repo")
        self.home = str(root / "home")
        self.out = str(root / "out")
        shutil.copytree(spec.local_repo, self.repo, symlinks=True)
        Path(self.home).mkdir()
        Path(self.out).mkdir()
        if not (Path(self.repo) / ".git").exists():
            # A plain fixture directory becomes a one-commit repository, so the
            # patch has a base commit exactly as a task image's checkout does.
            identity = [
                "-c", "user.name=swe-ab", "-c", "user.email=swe-ab@localhost",
                "-c", "core.excludesFile=/dev/null",
            ]  # fmt: skip
            for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
                _checked(self.run(["git", *identity, *argv]), f"git {argv[0]}")

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = _SETUP_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        # Setup runs under the same HOME as the agent: archex authenticates a
        # repo-local index with a machine secret kept under HOME, so an index
        # built under another HOME reads as `unprovenanced` to the hook.
        return subprocess.run(
            argv,
            cwd=cwd or self.repo,
            env=env if env is not None else {**os.environ, "HOME": self.home},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def put(self, source: Path, target: str) -> None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    def get(self, source: str, target: Path) -> None:
        if Path(source).resolve() != target.resolve():
            shutil.copy2(source, target)

    def close(self) -> None:
        return


class DockerRuntime:
    """One throwaway container of the task's official image (linux/amd64)."""

    def __init__(self, spec: CellSpec, *, mounts: list[tuple[Path, str]]) -> None:
        self.name = f"swe-ab-{uuid.uuid4().hex[:12]}"
        argv = [
            "docker",
            "run",
            "-d",
            "--platform",
            "linux/amd64",
            "--name",
            self.name,
            "--network",
            spec.network,
            "--entrypoint",
            "sleep",
        ]
        for host, target in mounts:
            argv += ["-v", f"{host}:{target}:ro"]
        _checked(
            subprocess.run(
                [*argv, spec.image, "infinity"], capture_output=True, text=True, check=False
            ),
            "docker run",
        )
        probe = self.run(["sh", "-c", "test -d /app && echo /app || echo /testbed"])
        self.repo = probe.stdout.strip() or "/app"
        self.home = "/root"
        self.out = "/out"
        _checked(self.run(["mkdir", "-p", self.out]), "mkdir /out")
        # Task images install their toolchains in image-specific places; the
        # agent keeps the image's own PATH, with archex's entry prepended only
        # in the arms that grant the CLI.
        self.image_path = self.run(["printenv", "PATH"], cwd="/").stdout.strip()

    def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = _SETUP_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        command = ["docker", "exec", "-w", cwd or getattr(self, "repo", "/")]
        for key, value in (env or {}).items():
            command += ["-e", f"{key}={value}"]
        return subprocess.run(
            [*command, self.name, *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def put(self, source: Path, target: str) -> None:
        _checked(self.run(["mkdir", "-p", str(Path(target).parent)], cwd="/"), f"mkdir {target}")
        _checked(
            subprocess.run(
                ["docker", "cp", str(source), f"{self.name}:{target}"],
                capture_output=True,
                text=True,
                check=False,
            ),
            f"docker cp {source}",
        )

    def get(self, source: str, target: Path) -> None:
        _checked(
            subprocess.run(
                ["docker", "cp", f"{self.name}:{source}", str(target)],
                capture_output=True,
                text=True,
                check=False,
            ),
            f"docker cp {source}",
        )

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, check=False)


@dataclass
class _Setup:
    base_commit: str = ""
    archex_version: str | None = None
    wheel_sha: str | None = None
    hook_sha: str | None = None
    guide_sha: str | None = None
    hook_path: str | None = None
    guide_path: str | None = None
    extra_path: list[str] = field(default_factory=list[str])
    setup_seconds: float = 0.0
    index_seconds: float | None = None


def _git(rt: Runtime, *args: str) -> str:
    return _checked(rt.run(["git", "-c", "core.excludesFile=/dev/null", *args]), f"git {args[0]}")


def _install_archex(spec: CellSpec, rt: Runtime) -> tuple[str, str | None]:
    """Interpreter archex runs from, and the installed wheel's SHA-256."""
    if spec.runtime == "local":
        return spec.archex_python, None
    if spec.archex_wheel is None or spec.uv_binary is None:
        raise ValueError("the docker runtime needs `archex_wheel` and `uv_binary`")
    wheel = f"/opt/archex/{spec.archex_wheel.name}"
    rt.put(spec.uv_binary, "/opt/archex/uv")
    rt.put(spec.archex_wheel, wheel)
    env = {"UV_PYTHON_INSTALL_DIR": "/opt/archex/python", "UV_CACHE_DIR": "/opt/archex/cache"}
    _checked(
        rt.run(
            ["/opt/archex/uv", "venv", "/opt/archex/venv", "--python", "3.12"], env=env, cwd="/"
        ),
        "uv venv",
    )
    _checked(
        rt.run(
            ["/opt/archex/uv", "pip", "install", "--python", "/opt/archex/venv/bin/python", wheel],
            env=env,
            cwd="/",
        ),
        "uv pip install archex",
    )
    return "/opt/archex/venv/bin/python", sha256_file(spec.archex_wheel)


def _prepare_archex(spec: CellSpec, rt: Runtime, setup: _Setup) -> None:
    python, setup.wheel_sha = _install_archex(spec, rt)
    setup.archex_version = _checked(
        rt.run([python, "-c", "import archex; print(archex.__version__)"]), "archex version"
    ).strip()
    exclude = f"{rt.repo}/.git/info/exclude"
    _checked(
        rt.run(["sh", "-c", f"mkdir -p $(dirname {exclude}) && echo '.archex/' >> {exclude}"]),
        "exclude .archex",
    )
    tracked = rt.run(["git", "ls-files", "--error-unmatch", ".gitignore"]).returncode == 0
    existed = rt.run(["test", "-e", ".gitignore"]).returncode == 0
    archex = [python, "-m", "archex.cli.main"]
    _checked(rt.run([*archex, "init", "--no-index", rt.repo]), "archex init")
    # `archex init` appends `.archex/` to `.gitignore`; the campaign never edits
    # a tracked `.gitignore`, and `.git/info/exclude` already covers it.
    if tracked:
        _git(rt, "checkout", "--", ".gitignore")
    elif not existed:
        rt.run(["rm", "-f", ".gitignore"])
    started = time.monotonic()
    _checked(rt.run([*archex, "index", rt.repo]), "archex index")
    setup.index_seconds = round(time.monotonic() - started, 3)
    status = json.loads(
        _checked(rt.run([*archex, "status", rt.repo, "--format", "json"]), "status")
    )
    if status.get("state") != "fresh":
        raise RuntimeError(f"index is {status.get('state')!r} after setup; the hook would be inert")
    if spec.arm.hook:
        module = spec.work_dir / "omp-annotate.ts"
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(render_annotation_hook_module(python), encoding="utf-8")
        setup.hook_sha = sha256_file(module)
        setup.hook_path = (
            f"{rt.out}/omp-annotate.ts"
            if spec.runtime == "local"
            else "/opt/archex/omp-annotate.ts"
        )
        rt.put(module, setup.hook_path)
    if spec.arm.cli:
        bin_dir = (
            f"{Path(rt.out).parent}/archex-bin" if spec.runtime == "local" else "/opt/archex/bin"
        )
        entry = str(Path(python).parent / "archex")
        _checked(
            rt.run(["sh", "-c", f"mkdir -p {bin_dir} && ln -sf {entry} {bin_dir}/archex"], cwd="/"),
            "archex on PATH",
        )
        setup.extra_path.append(bin_dir)


def _setup(spec: CellSpec, rt: Runtime) -> _Setup:
    setup = _Setup()
    started = time.monotonic()
    setup.base_commit = _git(rt, "rev-parse", "HEAD").strip()
    if spec.arm.archex_installed:
        _prepare_archex(spec, rt, setup)
    if spec.arm.cli:
        setup.guide_sha = sha256_file(spec.cli_guide)
        setup.guide_path = (
            f"{rt.out}/cli-guide.md" if spec.runtime == "local" else "/task/cli-guide.md"
        )
        rt.put(spec.cli_guide, setup.guide_path)
    rt.put(spec.profile_dir, f"{rt.home}/.omp/profiles/swebench/agent")
    prompt = spec.prompt_file or (spec.task_dir / "instruction.md" if spec.task_dir else None)
    if prompt is None:
        raise ValueError("the cell needs `prompt_file` or a task dir with instruction.md")
    rt.put(prompt, f"{rt.out}/prompt.md" if spec.runtime == "local" else "/task/prompt.md")
    setup.setup_seconds = round(time.monotonic() - started, 3)
    return setup


def _agent_env(spec: CellSpec, rt: Runtime, setup: _Setup) -> dict[str, str]:
    if spec.runtime == "local":
        # The agent's shell must not find an archex the arm does not grant.
        path = [
            entry
            for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry and not (Path(entry) / "archex").exists()
        ]
        base = {
            key: value for key, value in os.environ.items() if not key.startswith(("OMP_", "PI_"))
        }
    else:
        path = cast("DockerRuntime", rt).image_path.split(":")
        base = {}
    return {
        **base,
        "HOME": rt.home,
        "PATH": os.pathsep.join([*setup.extra_path, *path]),
        "ARCHEX_ANNOTATION_LEDGER": f"{rt.out}/annotation-ledger.jsonl",
        "ARCHEX_HOOK_DIAGNOSTICS_LOG": f"{rt.out}/hook-diagnostics.log",
    }


def _run_agent(spec: CellSpec, rt: Runtime, setup: _Setup, attempt: int) -> tuple[int | None, Path]:
    session_dir = f"{rt.out}/session-{attempt}"
    prompt = f"{rt.out}/prompt.md" if spec.runtime == "local" else "/task/prompt.md"
    argv = omp_argv(
        spec.omp_command,
        arm=spec.arm,
        model=spec.model,
        prompt_path=prompt,
        session_dir=session_dir,
        hook_module_path=setup.hook_path,
        cli_guide_path=setup.guide_path,
    )
    try:
        done = rt.run(
            argv, env=_agent_env(spec, rt, setup), timeout=_MAX_TIME_SECONDS + _OMP_GRACE_SECONDS
        )
        code: int | None = done.returncode
        (spec.work_dir / f"omp-stdout-{attempt}.jsonl").write_text(done.stdout, encoding="utf-8")
    except subprocess.TimeoutExpired:
        code = None
    local_sessions = spec.work_dir / f"session-{attempt}"
    if spec.runtime == "docker":
        rt.get(session_dir, local_sessions)
    else:
        local_sessions = Path(session_dir)
    return code, local_sessions


def _patch(rt: Runtime, base: str) -> str:
    _git(rt, "add", "-A")
    return _git(rt, "diff", "--cached", "--binary", base)


def index_in_container(spec: CellSpec, rt: Runtime) -> float:
    """Install archex into the running container and index the checkout (Stage 0)."""
    setup = _Setup()
    _prepare_archex(spec, rt, setup)
    return setup.index_seconds or 0.0


def score_patch(spec: CellSpec, patch: Path) -> bool:
    """Apply the patch in a fresh container and run the task's own verifier."""
    if spec.task_dir is None:
        raise ValueError("docker scoring needs `task_dir`")
    rt = DockerRuntime(spec, mounts=[])
    try:
        rt.put(spec.task_dir / "tests", "/tests")
        rt.put(patch, "/tmp/replay.patch")
        if patch.stat().st_size:
            rt.run(
                [
                    "sh",
                    "-c",
                    "git apply --verbose /tmp/replay.patch || git apply --3way "
                    "/tmp/replay.patch || patch --fuzz=3 -p1 -i /tmp/replay.patch",
                ]
            )
        rt.run(["bash", "/tests/test.sh"], timeout=_VERIFIER_TIMEOUT_SECONDS)
        reward = rt.run(["cat", "/logs/verifier/reward.txt"], cwd="/").stdout.strip()
        return reward == "1"
    finally:
        rt.close()


def _capture(spec: CellSpec) -> dict[str, Any] | None:
    if spec.capture_dir is None:
        return None
    requests = sorted(spec.capture_dir.glob("request-*.json"))
    return cast("dict[str, Any]", json.loads(requests[0].read_text())) if requests else None


def _isolation(spec: CellSpec, session: OmpSession) -> Isolation:
    first = _capture(spec)
    if first is not None:
        # Kept beside the cell (never committed): Stage 0 reads each arm's
        # rendered system prompt and tool list from it.
        (spec.work_dir / "first-request.json").write_text(json.dumps(first), encoding="utf-8")
        tools = sorted(
            str(cast("dict[str, Any]", tool.get("function") or {}).get("name"))
            for tool in cast("list[dict[str, Any]]", first.get("tools") or [])
        )
        messages = cast("list[dict[str, Any]]", first.get("messages") or [])
        system = "".join(
            str(message.get("content")) for message in messages if message.get("role") == "system"
        )
        violations = system_prompt_violations(system)
        checked = True
    else:
        tools, violations, checked = sorted(BASE_TOOLS), [], False
    texts = [exchange.result_text for exchange in session.exchanges]
    return Isolation(
        tools_source="request_capture" if checked else "declared",
        tools_advertised=tools,
        undeclared_tool_calls=sorted({e.tool for e in session.exchanges} - set(BASE_TOOLS)),
        system_prompt_checked=checked,
        system_prompt_violations=violations,
        compressor_marker_seen=any(m in text for text in texts for m in COMPRESSOR_MARKERS),
        annotation_seen=any(ANNOTATION_MARKER in text for text in texts),
        non_message_tokens=session.requests[0].non_message_tokens if session.requests else None,
    )


def failed_cell(spec: CellSpec, reason: FailureReason, detail: str, **identity: Any) -> SweAbCell:
    """A recorded failure: zero usage and counts, the reason, and whatever identity is known."""
    zero = dict.fromkeys(CHANNELS, 0)
    return SweAbCell(
        task_id=spec.task_id,
        repo=spec.repo,
        model=spec.model,
        arm=spec.arm,
        repetition=spec.repetition,
        status=CellStatus.FAILED,
        failure_reason=reason,
        failure_detail=detail[-500:],
        omp_version=OMP_VERSION,
        archex_version=identity.get("archex_version")
        or (None if not spec.arm.archex_installed else "unknown"),
        archex_wheel_sha256=identity.get("wheel_sha"),
        hook_module_sha256=identity.get("hook_sha") or ("unknown" if spec.arm.hook else None),
        cli_guide_sha256=identity.get("guide_sha") or ("unknown" if spec.arm.cli else None),
        image=spec.image,
        tool_fingerprint=tool_fingerprint(BASE_TOOLS),
        provider=None,
        provider_endpoint_overridden=provider_endpoint_overridden(spec.profile_dir, spec.model),
        usage=Usage(input=0, output=0, cache_read=0, cache_write=0, cost_usd=0.0),
        requests=0,
        tool_calls=0,
        channel_tokens_once=zero,
        channel_tokens_compounded=dict(zero),
        hook_ledger=HookLedgerSummary(results=0, eligible=0, annotated=0, units=0, tokens=0)
        if spec.arm.hook
        else None,
        archex_cli_calls=0,
        isolation=Isolation(
            tools_source="declared",
            tools_advertised=sorted(BASE_TOOLS),
            system_prompt_checked=False,
            compressor_marker_seen=False,
            annotation_seen=False,
        ),
        localization=Localization(gold_files=[]),
        patch_sha256="",
        patch_bytes=0,
        resolved=False,
        score_source="failed_before_patch",
        wall_seconds=0.0,
        setup_seconds=0.0,
    )


def run_cell(spec: CellSpec) -> SweAbCell:
    started = time.monotonic()
    if spec.capture_dir is not None:
        shutil.rmtree(spec.capture_dir, ignore_errors=True)
        spec.capture_dir.mkdir(parents=True)
    rt: Runtime
    if spec.runtime == "local":
        rt = LocalRuntime(spec)
    else:
        spec.work_dir.mkdir(parents=True, exist_ok=True)
        rt = DockerRuntime(spec, mounts=[(spec.omp_dir, "/opt/omp")] if spec.omp_dir else [])
    try:
        return _run_cell(spec, rt, started)
    finally:
        rt.close()


def _run_cell(spec: CellSpec, rt: Runtime, started: float) -> SweAbCell:
    try:
        setup = _setup(spec, rt)
    except Exception as exc:  # noqa: BLE001 - a setup failure is a recorded cell, never a crash
        return failed_cell(spec, FailureReason.HARNESS_ERROR, f"setup: {exc!r}")
    identity = {
        "archex_version": setup.archex_version,
        "wheel_sha": setup.wheel_sha,
        "hook_sha": setup.hook_sha,
        "guide_sha": setup.guide_sha,
    }

    retried = False
    code, session_dir = _run_agent(spec, rt, setup, attempt=1)
    session = _load_session(session_dir)
    if session is not None and session.error_message and not session.exchanges:
        retried = True  # provider failure before the first tool call: retry once, logged
        rt.run(["git", "checkout", "--", "."])
        code, session_dir = _run_agent(spec, rt, setup, attempt=2)
        session = _load_session(session_dir)
    if session is None:
        reason = FailureReason.TIMEOUT if code is None else FailureReason.SESSION_UNPARSABLE
        return failed_cell(spec, reason, f"omp exit {code}; no parsable session", **identity)

    patch_text = _patch(rt, setup.base_commit)
    patch_path = spec.work_dir / "model.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    patch_files = diff_files(patch_text)

    ledger_rows = read_ledger(_fetch(spec, rt, "annotation-ledger.jsonl"))
    edits = [e for e in session.exchanges if e.tool in ("edit", "write")]
    first_edit = edits[0].request_index if edits else None
    after_first_edit = {
        e.call_id
        for e in session.exchanges
        if first_edit is not None and e.request_index > first_edit
    }
    annotation_tokens = {
        str(row.get("toolCallId")): int(row.get("tokens") or 0)
        for row in ledger_rows
        if row.get("annotated")
    }
    obs = observations(session, annotation_tokens, count_tokens)
    once, compounded = compound(obs, len(session.requests))
    cli_calls = [
        sub
        for e in session.exchanges
        if e.tool == "bash"
        for sub in [archex_subcommand(str(e.arguments.get("command", "")))]
        if sub is not None
    ]
    gold = diff_files(spec.gold_patch.read_text()) if spec.gold_patch else []
    isolation = _isolation(spec, session)

    status, reason, detail = CellStatus.OK, None, None
    if code is None or (session.requests and session.requests[-1].stop_reason == "aborted"):
        status, reason, detail = (
            CellStatus.FAILED,
            FailureReason.TIMEOUT,
            f"omp exceeded {MAX_TIME}",
        )
    elif session.error_message:
        status, reason, detail = (
            CellStatus.FAILED,
            FailureReason.PROVIDER_ERROR,
            session.error_message,
        )
    elif (
        sorted(isolation.tools_advertised) != sorted(BASE_TOOLS)
        or isolation.undeclared_tool_calls
        or isolation.system_prompt_violations
        or isolation.compressor_marker_seen
        or (isolation.annotation_seen and not spec.arm.hook)
    ):
        status, reason, detail = (
            CellStatus.FAILED,
            FailureReason.FINGERPRINT_MISMATCH,
            isolation.model_dump_json(),
        )

    resolved: bool | None = None
    score_source: Literal["pro_verifier", "not_scored"] = "not_scored"
    if spec.runtime == "docker":
        resolved = score_patch(spec, patch_path)
        score_source = "pro_verifier"
    if status is CellStatus.FAILED:
        resolved = False

    requests = session.requests
    return SweAbCell(
        task_id=spec.task_id,
        repo=spec.repo,
        model=spec.model,
        arm=spec.arm,
        repetition=spec.repetition,
        status=status,
        failure_reason=reason,
        failure_detail=detail[-500:] if detail else None,
        retried=retried,
        omp_version=OMP_VERSION,
        archex_version=setup.archex_version,
        archex_wheel_sha256=setup.wheel_sha,
        hook_module_sha256=setup.hook_sha,
        cli_guide_sha256=setup.guide_sha,
        image=spec.image,
        tool_fingerprint=tool_fingerprint(isolation.tools_advertised),
        provider=requests[0].provider if requests else None,
        provider_endpoint_overridden=provider_endpoint_overridden(spec.profile_dir, spec.model),
        usage=Usage(
            input=sum(r.input for r in requests),
            output=sum(r.output for r in requests),
            cache_read=sum(r.cache_read for r in requests),
            cache_write=sum(r.cache_write for r in requests),
            cost_usd=round(sum(r.cost_usd for r in requests), 6),
        ),
        requests=len(requests),
        tool_calls=len(session.exchanges),
        tool_call_mix=dict(sorted(_count(e.tool for e in session.exchanges).items())),
        channel_tokens_once=once,
        channel_tokens_compounded=compounded,
        hook_ledger=summarize_ledger(ledger_rows, after_first_edit) if spec.arm.hook else None,
        archex_cli_calls=len(cli_calls),
        archex_cli_subcommands=dict(sorted(_count(cli_calls).items())),
        isolation=isolation,
        localization=localize(session, gold, patch_files, [rt.repo, "/app", "/testbed"]),
        patch_sha256=sha256_file(patch_path),
        patch_bytes=patch_path.stat().st_size,
        patch_files=patch_files,
        resolved=resolved,
        score_source=score_source,
        wall_seconds=round(time.monotonic() - started, 3),
        setup_seconds=setup.setup_seconds,
        index_seconds=setup.index_seconds,
    )


def _count(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item)] = counts.get(str(item), 0) + 1
    return counts


def _fetch(spec: CellSpec, rt: Runtime, name: str) -> Path:
    target = spec.work_dir / name
    if spec.runtime == "docker":
        if rt.run(["test", "-e", f"{rt.out}/{name}"], cwd="/").returncode == 0:
            rt.get(f"{rt.out}/{name}", target)
        return target
    return Path(rt.out) / name


def _load_session(session_dir: Path) -> OmpSession | None:
    files = sorted(session_dir.glob("*.jsonl")) if session_dir.is_dir() else []
    if len(files) != 1:
        return None
    try:
        return parse_omp_session(files[0])
    except (ValueError, OSError):
        return None


def main() -> int:
    spec = CellSpec(cast("dict[str, Any]", json.loads(sys.stdin.read())))
    try:
        cell = run_cell(spec)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, nothing dropped
        cell = failed_cell(spec, FailureReason.HARNESS_ERROR, repr(exc))
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    spec.output.write_text(cell.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"cell": str(spec.output), "status": cell.status, "reason": cell.failure_reason})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
