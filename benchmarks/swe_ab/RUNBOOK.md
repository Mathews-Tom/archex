# SWE A/B campaign runbook (Stage 0 onward)

Operator steps for the SWE-task A/B of archex surfaces under omp. The protocol is `benchmarks/preregistrations/R3x-swe-archex-ab.md`; this file only says how to run it. Nothing here spends money until §4, and §4 needs the maintainer's explicit go-ahead.

## 1. Host

SWE-bench Pro images are `linux/amd64`, and scoring runs each task's own test suite. Emulated runs on an arm64 workstation are slow and can fail spuriously, which would corrupt solve rate. Use:

| Resource | Requirement |
| --- | --- |
| Architecture / OS | x86_64 Linux (Ubuntu 22.04+ or Debian 12), Docker Engine 24+ with the daemon running |
| CPU | ~32 vCPU (cells run in parallel per provider, up to the provider's rate limit) |
| Memory | 128 GB |
| Disk | 1 TB (Pro V2 images are pulled per instance) |
| Network | egress to `ghcr.io`, the model provider endpoints, and PyPI/GitHub for setup |
| Tools | `git`, `uv` (host), `jq`, `bun` 1.x (to build the omp bundle) |

## 2. One-time setup

```bash
git clone https://github.com/Mathews-Tom/archex && cd archex
uv sync --all-extras

# SWE-bench Pro V2 tasks (instruction.md, tests/, solution/ per instance)
git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os ../SWE-bench_Pro-os
TASKS=../SWE-bench_Pro-os/v2/tasks
(cd ../SWE-bench_Pro-os/v2 && shasum -a 256 -c SHA256SUMS >/dev/null && echo "tasks verified")

# omp 18.4.2 as a linux-x64 bundle mounted read-only at /opt/omp in every task container
mkdir -p /opt/omp-linux-x64
docker run --rm --platform linux/amd64 -v /opt/omp-linux-x64:/opt/omp -e BUN_INSTALL=/opt/omp \
  oven/bun:1-debian sh -c 'bun install -g @oh-my-pi/pi-coding-agent@18.4.2 && cp "$(command -v bun)" /opt/omp/bin/bun'
# The entry point inside containers is then:
OMP_COMMAND="/opt/omp/bin/bun /opt/omp/install/global/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js"

# Static uv for installing archex inside each container
mkdir -p /opt/uv && curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz -C /opt/uv --strip-components=1

# The archex release that ships the hook (pinned by wheel SHA-256 in every cell)
uv build --wheel && ARCHEX_WHEEL=$(ls dist/archex-*-py3-none-any.whl | tail -1)
sha256sum "$ARCHEX_WHEEL"
```

Provision the isolated profile once, with auto-update off, no memory, no advisor, and credentials for the campaign's provider route (the Stage 0 catalog check resolved all four models to `amazon-bedrock`):

```bash
omp --profile swebench            # log in / configure the provider, then quit
PROFILE_DIR=~/.omp/profiles/swebench/agent
ls "$PROFILE_DIR"                 # config.yml, agent.db, …; no extensions/, skills/, rules/, AGENTS.md
```

The harness copies the profile into each container per cell, so the host copy stays read-only. It must hold no `models.yml` that overrides a campaign provider's `baseUrl`: a cell whose model resolves through a profile-declared endpoint records `provider_endpoint_overridden: true` and the validator refuses it.

## 3. Stage 0 checks (no model spend)

```bash
# Instances for the validity check: the frozen Stage 1 sample, one id per line
printf '%s\n' <instance_id> ... > stage0-instances.txt

uv run python scripts/swe_ab_stage0.py --output benchmarks/swe_ab/stage0.json --host \
  --tasks-root "$TASKS" --instances stage0-instances.txt \
  --omp-dir /opt/omp-linux-x64 --omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary /opt/uv/uv
jq '.summary, .gate, [.checks[] | select(.status != "pass") | {id, status, detail}]' benchmarks/swe_ab/stage0.json
```

Without `--host` the script runs only the local checks: omp version, model-id resolution, frozen identities, and one no-spend cell per arm against the local stub provider (tool list, system-prompt isolation, search-routing disclosure, annotation only in H/HC, no compressor output, validator refusal). With `--host` it adds, per instance: gold patch resolves and empty patch fails, the omp build starts in the image, and archex installs into `/opt/archex` and indexes the checkout to `fresh`.

Instances failing gold/empty validity leave the candidate pool before any agent run; list them in the pre-registration.

## 4. One real cell per model (paid — explicit go-ahead required)

This is the last Stage 0 check. One short task, arm `H` (it exercises the route, auth, usage reporting, and the annotation ledger), one repetition per model:

```json
{
  "name": "stage0-one-cell",
  "tasks": [{"task_id": "<one short instance_id>", "repo": "<org/repo>"}],
  "models": ["global.anthropic.claude-sonnet-5-5", "global.anthropic.claude-opus-5-5",
             "global.openai.gpt-6-sol", "global.openai.gpt-6-luna"],
  "repetitions": {"H": 1},
  "cost_ceiling_usd": 40
}
```

```bash
uv run python scripts/run_swe_ab_suite.py run --plan stage0-one-cell.json --runtime docker \
  --output benchmarks/swe_ab/results/stage0 --work-root /scratch/swe-ab/stage0 \
  --tasks-root "$TASKS" --omp-dir /opt/omp-linux-x64 --omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary /opt/uv/uv --jobs 4
uv run python scripts/run_swe_ab_suite.py validate --plan stage0-one-cell.json \
  --input benchmarks/swe_ab/results/stage0
jq '{model, status, failure_reason, usage, hook: .hook_ledger, provider}' \
  benchmarks/swe_ab/results/stage0/*/H/*.json
```

Pass: every cell `ok`, `usage` non-zero at every tier the provider reports, `provider` is the expected route, `hook_ledger.annotated > 0` when the agent searched before editing.

## 5. Cost ceiling flags

- The plan's `cost_ceiling_usd` is the hard cumulative ceiling for the stage. `--cost-ceiling` can only lower it.
- The suite checks, before starting each cell, that spent cost plus the largest recorded cell cost for every cell still running stays under the ceiling; otherwise it waits for running cells, records them, and exits with status 3.
- Recorded cost is omp's per-request `usage.cost.total`, summed per cell. Resumed runs count existing cells' cost toward the ceiling.
- `--jobs N` runs N cells in parallel; keep N within the provider's rate limit.

## 6. Stage 1 and Stage 2

Freeze the pre-registration first (fill every *(set at freeze)* field), then run each stage with its plan exactly as in §4 with `--runtime docker`, and validate the result directory before any analysis. The run resumes by skipping cells whose artifact exists. A harness defect found after data exists means discarding and restarting the affected stage, never repairing cells.

## 7. No-spend rehearsal (any machine with omp and archex)

```bash
uv run python scripts/run_swe_ab_suite.py run --plan benchmarks/swe_ab/dry-run-plan.json \
  --runtime local --output /tmp/swe-ab/cells --work-root /tmp/swe-ab/work \
  --omp-command "$(command -v omp)" --stub-script benchmarks/swe_ab/stub-script.json
uv run python scripts/run_swe_ab_suite.py validate --plan benchmarks/swe_ab/dry-run-plan.json \
  --input /tmp/swe-ab/cells     # exits 1: stub-endpoint cells are refused for publication
```

The stub (`scripts/swe_ab_stub_provider.py`) replays a `grep`, a `read`, an `edit`, and a stop over the OpenAI chat-completions SSE protocol; each cell runs with an isolated `HOME` and its own copy of the fixture repository.
