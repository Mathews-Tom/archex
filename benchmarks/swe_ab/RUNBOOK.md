# SWE A/B campaign runbook (Stage 0 onward)

Operator steps for the SWE-task A/B of archex surfaces under omp. The protocol is `benchmarks/preregistrations/R3x-swe-archex-ab.md`; this file only says how to run it. Nothing here spends subscription quota until §5, and §5 needs the maintainer's explicit go-ahead and the pre-registration's terms-of-service item closed.

The campaign runs on the operator's **Claude and ChatGPT subscriptions** (`anthropic/claude-sonnet-5-5`, `anthropic/claude-opus-5-5`, `openai-codex/gpt-6-sol`, `openai-codex/gpt-6-luna`). omp holds those logins in its own `agent.db`; the agent containers never receive that file. They reach the logins through omp's **auth broker**, which runs on this host.

## 1. Host

SWE-bench Pro images are `linux/amd64`, and scoring runs each task's own test suite. Two hosts are supported; **a stage must run entirely on one kind**, because emulated wall times, timeouts, and (rarely) test outcomes differ. Every cell records `emulated: true|false`, and the validator refuses a stage that mixes them.

| | x86_64 Linux (campaign) | Apple silicon Mac (local, emulated) |
| --- | --- | --- |
| Runtime | Docker Engine 24+, daemon running | Docker Desktop with x86_64/amd64 emulation through Rosetta |
| CPU | ~32 vCPU, `--jobs` up to the provider limit | give Docker at most 80% of the cores; `--jobs 1` or `2` |
| Memory | 128 GB | at most ~60% of RAM to Docker (a 32 GB Mac: 20 GiB); agent + verifier run one container at a time per job |
| Disk | 1 TB | Docker disk image ≥ 150 GiB free (Pro images are multi-GB each); see §1.2 |
| Network | egress to `ghcr.io`, the two provider endpoints, PyPI/GitHub for setup; the host broker | same; containers reach the broker as `host.docker.internal` |
| Tools | `git`, `uv`, `jq`, `bun` 1.x | same, plus Docker Desktop 4.x |

### 1.1 Docker Desktop with Rosetta (Apple silicon)

1. Docker Desktop → Settings → General: enable **Use Rosetta for x86_64/amd64 emulation on Apple Silicon** (needs the Apple Virtualization framework). Without it Docker falls back to QEMU, which is slower and less faithful; do not mix it into a stage.
2. Settings → Resources: set CPUs, Memory, and Disk image size as in the table. Apply & restart.
3. Start Docker Desktop **before** any `--host` step. Stage 0 reports every container check as `requires_host` when `docker info` fails; it never starts Docker for you.
4. The omp bundle and uv must sit under a path Docker Desktop shares with its VM (your home directory, not `/opt`). The commands below use `WORK`:

```bash
export WORK=/opt              # x86_64 Linux
export WORK="$HOME/swe-ab"    # Apple silicon: a shared path
mkdir -p "$WORK"
```

Emulation facts to keep in mind (Stage 0 records the numbers for the machine in use, `emulated_wall_times`):

- Every image is started with `--platform linux/amd64`; the cell's `emulated` field is `true` on any arm64 host.
- Time caps are wall-clock: the 60-minute omp cap and the 3000 s verifier cap bind sooner under emulation. Emulated timeouts are recorded failures like any other, so an emulated stage is not comparable with a native one; it is only comparable with itself.
- Whether an instance's tests pass under emulation is itself checked: an instance whose gold patch does not resolve, or whose empty patch does not fail, under this host's emulation leaves the pool (§4).

### 1.2 Disk hygiene

Each Pro image is several GB and a stage touches dozens to hundreds. Pass `--prune-images` to the suite so it removes a task's image after every cell of that task has run (cells of one task are contiguous in the plan). Otherwise:

```bash
docker image rm ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>     # after that task's last cell
docker system df                                                    # check what is left
docker builder prune -f && docker image prune -f                    # dangling layers
```

Containers are already throwaway: the cell runner `docker rm -f`s each one, including the scoring container.

## 2. One-time setup

```bash
git clone https://github.com/Mathews-Tom/archex && cd archex
uv sync --all-extras

# SWE-bench Pro V2 tasks (instruction.md, tests/, solution/ per instance)
git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os ../SWE-bench_Pro-os
TASKS=../SWE-bench_Pro-os/v2/tasks
(cd ../SWE-bench_Pro-os/v2 && shasum -a 256 -c SHA256SUMS >/dev/null && echo "tasks verified")

# omp 18.4.4 as a linux-x64 bundle mounted read-only at /opt/omp in every task container
mkdir -p "$WORK/omp-linux-x64"
docker run --rm --platform linux/amd64 -v "$WORK/omp-linux-x64:/opt/omp" -e BUN_INSTALL=/opt/omp \
  oven/bun:1-debian sh -c 'bun install -g @oh-my-pi/pi-coding-agent@18.4.4 && mkdir -p /opt/omp/bin && cp "$(command -v bun)" /opt/omp/bin/bun'
# The entry point inside containers is then:
OMP_COMMAND="/opt/omp/bin/bun /opt/omp/install/global/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js"

# Static uv for installing archex inside each container
mkdir -p "$WORK/uv" && curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz -C "$WORK/uv" --strip-components=1

# The archex release that ships the hook (pinned by wheel SHA-256 in every cell)
uv build --wheel && ARCHEX_WHEEL=$(ls dist/archex-*-py3-none-any.whl | tail -1)
sha256sum "$ARCHEX_WHEEL"
```

### 2.1 Bun under emulation

The install above uses Bun's default x86-64 build, which needs AVX2. Whether Rosetta for Linux exposes AVX2 to it is **checked by Stage 0** (`bun_runs_under_emulation`, §4), not assumed:

```bash
docker run --rm --platform linux/amd64 oven/bun:1-debian bun --version   # "Illegal instruction" => no AVX2
```

If the default build fails, use Bun's **baseline** build (no AVX2 needed) for the bundle: replace the copied binary with `bun-linux-x64-baseline` from the Bun release page.

```bash
docker run --rm --platform linux/amd64 -v "$WORK/omp-linux-x64:/opt/omp" debian:12-slim sh -c \
  'apt-get update -qq && apt-get install -y -qq curl unzip ca-certificates >/dev/null &&
   curl -fsSL https://github.com/oven-sh/bun/releases/latest/download/bun-linux-x64-baseline.zip -o /tmp/b.zip &&
   unzip -q /tmp/b.zip -d /tmp && cp /tmp/bun-linux-x64-baseline/bun /opt/omp/bin/bun && /opt/omp/bin/bun --version'
```

Bun's own installer takes the same decision: it requests the `-baseline` build when `/proc/cpuinfo` shows no `avx2` (`https://bun.sh/install`, the `avx2` check). omp's native addon needs no action: it detects AVX2 at load time and falls back to its baseline addon (`@oh-my-pi/pi-natives/native/loader-state.js`, variant `modern` vs `baseline`; `PI_NATIVE_VARIANT=baseline` forces it). The emulated omp start is timed by the Stage 0 `omp_runs_in_container` and `emulated_wall_times` checks.

### 2.2 The profile holds no logins

Provision the isolated profile once, with auto-update off, no memory, and no advisor, **without logging in**:

```bash
omp --profile swebench            # configure only, then quit; do not run /login
PROFILE_DIR=~/.omp/profiles/swebench/agent
rm -f "$PROFILE_DIR"/agent.db "$PROFILE_DIR"/agent.db-*    # omp creates its SQLite store here; it is also the login vault
ls "$PROFILE_DIR"                 # config.yml, …; no agent.db, no extensions/, skills/, rules/, AGENTS.md
```

The harness copies the profile into each container per cell. It **refuses** a profile that carries `agent.db*`, a `*.token` file, or an encrypted broker snapshot (`*.enc`): the suite exits before any cell, and the cell runner records a `harness_error` if it is reached anyway. Credentials never travel as a copied file. The profile must also hold no `models.yml` that overrides a campaign provider's `baseUrl`: a cell whose model resolves through a profile-declared endpoint records `provider_endpoint_overridden: true` and the validator refuses it.

## 3. The auth broker: credentials for the containers

`omp auth-broker serve` runs on this host over omp's existing `agent.db` and is the only process that refreshes the logins. It answers on `http://<bind>/v1`; every route except `/v1/healthz` needs the bearer token in `<omp config dir>/auth-broker.token` (mode 0600). Source: omp's `docs/auth-broker-gateway.md` (`omp://auth-broker-gateway.md`).

### 3.1 Start it

```bash
# Docker Desktop (macOS): loopback is enough; containers reach it as host.docker.internal.
omp auth-broker serve --bind 127.0.0.1:8765

# Linux host: bind the docker bridge gateway, never 0.0.0.0.
GATEWAY=$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}')   # usually 172.17.0.1
omp auth-broker serve --bind "$GATEWAY:8765"
```

Leave it running for the whole stage. Do not print the token: the suite reads it from `--broker-token-file` (default `<omp config dir>/auth-broker.token`, or `$OMP_AUTH_BROKER_TOKEN` if set), and passes it to each agent container as the environment variable `OMP_AUTH_BROKER_TOKEN`. The broker URL the containers see is `http://host.docker.internal:<port>` (`--container-broker-url` overrides it). On Linux, containers need the mapping `--add-host host.docker.internal:host-gateway`; the cell runner adds it to every campaign container (harmless on Docker Desktop, where the name already resolves).

**Reachability was not verified from a Docker Desktop container in the session that wrote this** (Docker Desktop was installed but not running). The Stage 0 check `broker_reachable_from_container` does it on the operator's machine: a throwaway `alpine` container with the `--add-host` mapping fetches `/v1/healthz` and reports the bind address you passed as `--broker-bind`. If a loopback-bound broker is unreachable from the container, do **not** rebind to `0.0.0.0` (that exposes the vault's API to the LAN): bind the Docker bridge/VM gateway address instead, or stop and ask.

### 3.2 What the container receives, and what that exposes

An explicit allow-list: **two variables**, `OMP_AUTH_BROKER_URL` and `OMP_AUTH_BROKER_TOKEN`, and nothing else from the host environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, cloud credentials, and the rest are never forwarded). Each cell records the variable **names** in `credential_env_names`, never the values; the validator refuses a cell whose names are not exactly those two. The token is passed as a bare `-e OMP_AUTH_BROKER_TOKEN` with its value in the docker client's environment, so it does not appear in any process listing or argv. `agent.db` is never copied into a container (§2.2).

Be clear about what that exposes. The agent runs with `--approval-mode yolo` and a `bash` tool, so **the agent can read `OMP_AUTH_BROKER_TOKEN` from its own environment**. With it, the agent can call the broker: read `/v1/snapshot` (access tokens for the logins, with refresh tokens replaced by a sentinel), disable or block credentials, and read usage. Consequences:

- Access tokens are short-lived and refresh tokens are never sent to clients, so a leak is bounded in time, but the agent could burn or exfiltrate quota while its token is valid, and could disable a login (the run then stops on provider errors).
- Rotate the token after every stage: `omp auth-broker token --regenerate` (this invalidates running containers' token, so do it between stages).
- The broker's `agent.db` holds the operator's real subscription logins. Run the campaign only on a login the operator is content to expose this way.
- omp also offers `omp auth-gateway serve`, where clients never see an access token because the gateway proxies the model requests. It is **not** used here: it re-encodes requests through a different endpoint, which the validator treats as an overridden provider endpoint and which would change what is measured.

### 3.3 Egress

The agent container should reach only the broker and the two provider endpoints, but that is **not enforced here**, and cannot be reduced to "the broker only": in broker mode the container's omp calls Anthropic and OpenAI directly with the access token from the snapshot. The V2 protocol runs the agent offline except the model endpoint; this harness's default is Docker's `bridge` network with open egress, and the choice is fixed at freeze and recorded per cell (`network`). A restrictive setup that is feasible but untested here:

- Linux: create a network (`docker network create swe-ab`), run the suite with `--network swe-ab`, and add `iptables -I DOCKER-USER` rules for that subnet: allow the broker address:port, DNS, and the provider hosts (`api.anthropic.com`, `chatgpt.com`), drop everything else.
- Docker Desktop offers no host firewall for container traffic; the only equivalent is an `--internal` network plus an allow-listing HTTPS proxy container, which needs proxy variables in the allow-list above. Not implemented.

## 4. Stage 0 checks (no model spend)

```bash
# Instances for the validity check: the frozen Stage 1 sample, one id per line
printf '%s\n' <instance_id> ... > stage0-instances.txt

uv run python scripts/swe_ab_stage0.py --output benchmarks/swe_ab/stage0.json --host \
  --tasks-root "$TASKS" --instances stage0-instances.txt \
  --omp-dir "$WORK/omp-linux-x64" --omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary "$WORK/uv/uv" \
  --broker-url http://127.0.0.1:8765 --broker-bind 127.0.0.1:8765
jq '.summary, .gate, [.checks[] | select(.status != "pass") | {id, status, detail}]' benchmarks/swe_ab/stage0.json
```

Without `--host` (or when `docker info` fails) the script runs only the local checks and leaves the container checks `requires_host`:

| Check | What it establishes | Model call |
| --- | --- | --- |
| `omp_version` | the installed omp is the pinned build | no |
| `model_routes_and_logins` | each campaign id resolves in omp's catalog to its own subscription provider (`anthropic` or `openai-codex`, not Bedrock or a router) and that provider has an enabled login; reads `omp models --json` and login metadata from `agent.db` (provider, type, disabled flag; never the credential payload) | no |
| `frozen_identities` | SHA-256 of the CLI guide and the rendered hook module | no |
| `broker_healthy`, `broker_usage_readable` | the broker answers `/v1/healthz`, and its `/v1/usage` is readable, per-model headroom as the quota guard sees it (`unknown` if the broker reports no usage for a provider) | no |
| `stub_rehearsal` … `validator_refuses_stub_cells` | one no-spend cell per arm against the local stub: tool list, system-prompt isolation, search-routing disclosure, annotation only in H/HC, no compressor output, validator refusal of stub-endpoint cells | no |
| `gold_empty_validity` | per instance, the gold patch resolves and the empty patch fails (invalid instances leave the pool) | no |
| `omp_runs_in_container` | the pinned omp build starts inside each Pro image | no |
| `archex_indexes_in_container` | archex installs into `/opt/archex`, indexes the checkout to `fresh`, and the annotate entry is pre-warmed on a real search hit | no |
| `emulated_wall_times` | container start, gold/empty scoring, omp start, and install+index seconds per instance; `emulated` and the container architecture | no |
| `broker_reachable_from_container` | a container reaches the host broker and receives only the two allow-listed variables | no |
| `bun_runs_under_emulation` | Bun's default x86-64 build starts under emulation, or the baseline build does (§2.1) | no |
| `one_real_cell_per_model` | §5 | **yes** |

The annotate **pre-warm** runs in every hook-arm cell's setup: `python -m archex.integrations.annotate_hook` is run once on a real `git grep` hit in the checkout, so the first call the agent sees is not the cold start (bytecode compilation, tokenizer load, cold caches) that could exceed the hook's 0.5 s budget and silently drop an annotation. It logs to `/dev/null`, must reach the search path (or the cell fails at setup), and its duration is recorded in `annotate_prewarm_seconds`.

Instances failing gold/empty validity leave the candidate pool before any agent run; list them in the pre-registration.

## 5. One real cell per model (uses subscription quota — explicit go-ahead required)

This is the last Stage 0 check. **Do not run it until the pre-registration's terms-of-service item is closed and the maintainer says go.** One short task, arm `H` (it exercises the route, auth, usage reporting, and the annotation ledger), one repetition per model. It consumes a small amount of each subscription's rolling window; `cost_ceiling_usd` is omp's list-price model of the tokens, a runaway guard, not a bill.

```json
{
  "name": "stage0-one-cell",
  "tasks": [{"task_id": "<one short instance_id>", "repo": "<org/repo>"}],
  "models": ["anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5",
             "openai-codex/gpt-6-sol", "openai-codex/gpt-6-luna"],
  "repetitions": {"H": 1},
  "cost_ceiling_usd": 40
}
```

```bash
omp auth-broker serve --bind 127.0.0.1:8765 &          # or the Linux bind in §3.1
uv run python scripts/run_swe_ab_suite.py run --plan stage0-one-cell.json --runtime docker \
  --output benchmarks/swe_ab/results/stage0 --work-root "$WORK/stage0" \
  --tasks-root "$TASKS" --omp-dir "$WORK/omp-linux-x64" --omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary "$WORK/uv/uv" \
  --broker-url http://127.0.0.1:8765 --prune-images --jobs 1
uv run python scripts/run_swe_ab_suite.py validate --plan stage0-one-cell.json \
  --input benchmarks/swe_ab/results/stage0
jq '{model, status, failure_reason, usage, hook: .hook_ledger, provider, emulated, credential_env_names, quota}' \
  benchmarks/swe_ab/results/stage0/*/H/*.json
```

Pass: every cell `ok`, `usage` non-zero at every tier the provider reports, `provider` is `anthropic` or `openai-codex` matching the model's prefix, `credential_env_names` is the two broker variables, `hook_ledger.annotated > 0` when the agent searched before editing. Record `quota` (omp retries, block phase) for each model.

## 6. Quota and rate limits

Subscriptions have rolling windows (a 5-hour window and a weekly one) instead of a bill. What omp does when one is hit, from the pinned build's source (`@oh-my-pi/pi-coding-agent` 18.4.4, `src/session/turn-recovery.ts` unless noted; `@oh-my-pi/pi-ai` for the rest):

- **It retries, rotates, waits, or fails fast, depending on the reset time.** A rate-limit or usage-limit error is classified (`pi-ai/src/error/flags.ts:520-550`, `rate-limit.ts`) and handed to `#handleRetryableError` (`turn-recovery.ts:2350`). The usage-limit outcome is recorded first (`recordUsageLimitOutcome`, `:732`), which marks the credential blocked until the provider-stated reset (`pi-ai/src/auth/rotation.ts:229` `markReached`).
- **Rotate:** with a sibling login for the same provider, omp switches to it at once (`:2434-2473`, `rotate` at `rotation.ts:396`). The campaign has one login per provider, so this rarely applies.
- **Wait:** otherwise it sleeps until the credential's reset (`sleepLong`, `:2773`) — but only if the wait is at most `retry.maxDelayMs`, default **5 minutes** (`session/settings.ts:689-700`). Transient caps use reason-specific backoffs: 30 s for a rate limit, 5 s for a concurrency cap, 30 min assumed for a quota with no stated reset (`pi-ai/src/error/rate-limit.ts:17-19`).
- **Fail fast:** a longer wait ends the turn with the provider error (`:2702-2740`, `Provider requested Nms wait, exceeds retry.maxDelayMs`) unless `retry.waitForUsageReset` is on. It is **off** by default (`settings.ts:702`) and the harness leaves it off: the wait would hold a cell past its 60-minute cap. The retry budget is `retry.maxRetries` = 10 (`settings.ts:668`); past it the last error surfaces (`Retry budget exhausted after 10 retries`).
- **A blocked single credential is still tried.** The credential ranking has a last-resort pass that allows blocked accounts (`pi-ai/src/auth/select.ts:816-824`), so a stale block never wedges a run; the provider's own answer decides.
- omp does not distinguish "before the first tool call" from "mid-run"; the harness does.

Observed against omp 18.4.4 with the local stub answering HTTP 429 (`scripts/swe_ab_stub_provider.py`, the `http_error` step; no hosted call): a usage-limit error with a 4-hour reset ended the run at once with `Provider requested 14400000ms wait, exceeds retry.maxDelayMs (300000ms)`, both before the first tool call and after a first tool call; a rate limit with a 1-second reset made omp retry 10 times with backoff (about 47 s of waiting in the stream's `auto_retry_start` events) and then fail with `Retry budget exhausted after 10 retries`. Both are recorded as `quota_block`.

### What the harness does

- **Before each cell attempt** the suite reads the broker's `GET /v1/usage` and checks the cell's provider. If every login has less than `--quota-min-headroom` (default 10%) left on a window that applies to the model, it pauses, re-checking every `--quota-poll-seconds` (default 60 s) and waking just after the reported reset. A broker that cannot be reached is waited out, never treated as headroom. A provider for which the broker reports nothing usable (`unknown`) proceeds, noted once in the output.
- **A cell that ends in a quota block** (a final rate-limit or quota error, before its first tool call or mid-run) is **not a task outcome**. It is recorded with `failure_reason: quota_block` and `quota.block_phase` of `before_first_tool_call` or `mid_run`, its tokens counted in cost, and its artifact filed under `<output>/quota-blocked/` (never scored). The suite waits for the block to clear (an initial cooling-down of 5 minutes, then the guard) and re-runs the cell, up to `--quota-retries` (default 2) per invocation. omp's own in-run retries and waits are recorded in `quota.omp_retries` and `quota.omp_retry_wait_seconds`.
- **Provider failures that are not quota blocks** keep the earlier rule: a failure before the first tool call is retried once, logged (`retried: true`); anything later is a recorded `provider_error`.
- **If the quota does not clear within `--quota-max-wait-seconds`** (default 6 h), the run stops with **exit status 4**. It is resumable: run the same command again. A cell whose retries ran out is filed as a blocked attempt and re-run on resume.
- **Publication:** `validate` accepts a directory only when no cell at its cell path is a quota block, and reports the number of filed blocked attempts (`quota_blocked`). Blocked attempts are disclosed in the analysis per arm.
- The modelled-USD ceiling (§7) stays as the runaway guard; the quota guard is what keeps the run from burning cells against an empty window.

Quota rehearsal, no spend (each block is classified by a real omp run against the stub):

```bash
cat > /tmp/quota-stub.json <<'EOF'
[{"tool": "grep", "args": {"i": "Locating the hash helper", "pattern": "hash_password"}},
 {"http_error": 429, "message": "You have hit your usage limit. Your limit will reset in 4 hours.", "retry_after": 14400}]
EOF
uv run python scripts/run_swe_ab_suite.py run --plan benchmarks/swe_ab/dry-run-plan.json \
  --runtime local --output /tmp/swe-ab/quota-cells --work-root /tmp/swe-ab/quota-work \
  --omp-command "$(command -v omp)" --stub-script /tmp/quota-stub.json
# every arm's cell is recorded with failure_reason quota_block, phase mid_run
```

## 7. Cost ceiling flags

- The plan's `cost_ceiling_usd` is the hard cumulative ceiling for the stage. `--cost-ceiling` can only lower it.
- The suite checks, before starting each cell, that spent cost plus the largest recorded cell cost for every cell still running stays under the ceiling; otherwise it waits for running cells, records them, and exits with status 3.
- Recorded cost is omp's per-request `usage.cost.total` — a list-price model of the tokens — summed per cell, **including quota-blocked attempts**. On a subscription it is not a bill. Resumed runs count existing cells' cost toward the ceiling.
- `--jobs N` runs N cells in parallel; keep N within the provider's rate limit and the host's resources (§1).

## 8. Stage 1 and Stage 2

Freeze the pre-registration first (fill every *(set at freeze)* field), then run each stage with its plan exactly as in §5 with `--runtime docker`, and validate the result directory before any analysis. The run resumes by skipping cells whose artifact exists. A harness defect found after data exists means discarding and restarting the affected stage, never repairing cells. Rotate the broker token between stages (§3.2).

## 9. No-spend rehearsal (any machine with omp and archex)

```bash
uv run python scripts/run_swe_ab_suite.py run --plan benchmarks/swe_ab/dry-run-plan.json \
  --runtime local --output /tmp/swe-ab/cells --work-root /tmp/swe-ab/work \
  --omp-command "$(command -v omp)" --stub-script benchmarks/swe_ab/stub-script.json
uv run python scripts/run_swe_ab_suite.py validate --plan benchmarks/swe_ab/dry-run-plan.json \
  --input /tmp/swe-ab/cells     # exits 1: stub-endpoint cells are refused for publication
```

The stub (`scripts/swe_ab_stub_provider.py`) replays a `grep`, a `read`, an `edit`, and a stop over the OpenAI chat-completions SSE protocol; each cell runs with an isolated `HOME` and its own copy of the fixture repository. The local runtime carries no broker credentials and records `emulated: false`, `credential_env_names: []`.
