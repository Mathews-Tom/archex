# SWE A/B campaign runbook (Stage 0 onward)

Operator steps for the SWE-task A/B of archex surfaces under omp. The protocol is `benchmarks/preregistrations/R3x-swe-archex-ab.md`; this file only says how to run it. Nothing here spends credits until §5, and §5 needs the maintainer's explicit go-ahead.

## Campaigns: the provider is data, not code

A **campaign** file under `benchmarks/swe_ab/campaigns/` selects the provider and the configurations; every plan names one in `campaign`, and the stage-0 script and the sampler take `--campaign`. A *configuration* is a model at a thinking effort. The campaign points at a frozen omp `models.yml` (`provider_config`) that holds **exactly one provider**: its `baseUrl`, `api`, `apiKey` (the *name* of the environment variable carrying the key; the only credential an agent container receives), the models, and their prices. omp's `--model` (`<provider>/<model id>`) and `--thinking` are derived from the campaign, never typed. Switching to another provider or model (a new stealth or free preview) is a new provider config plus a new campaign file; nothing in the harness names a provider.

| Campaign | Provider config | Key variable | Configurations (plan/cell `model` → omp selector, `--thinking`) |
| --- | --- | --- | --- |
| `campaigns/openrouter.yml` (**draft**) | `openrouter-open-models.yml` (`https://openrouter.ai/api/v1`, priced; each model pinned to one upstream host) | `OPENROUTER_API_KEY` | `qwen-3.8-27b@low` → `openrouter/qwen/qwen3.8-27b`, `low` (DeepInfra bf16); `qwen-3.8-27b@high` → same, `high`; `gemma-4-26b-a4b-it@high` → `openrouter/google/gemma-4-26b-a4b-it`, `high` (NextBit bf16) |
| `campaigns/muna.yml` | `muna-models.yml` (`https://inference.muna.ai/v1`, priced) | `MUNA_ACCESS_KEY` | `qwen-3.8-27b@low` → `muna/@qwen/qwen-3.8-27b`, `low`; `qwen-3.8-27b@high` → same, `high`; `gemma-4-26b-a4b-it@high` → `muna/@google/gemma-4-26b-a4b-it`, `high` |
| `campaigns/openrouter-space-bunny.yml` | `openrouter-models.yml` (`https://openrouter.ai/api/v1`, free) | `OPENROUTER_API_KEY` | `space-bunny-alpha@high` → `openrouter/stealth/space-bunny-alpha`, `high` |

**Pinning an OpenRouter host.** OpenRouter spreads one model id over several hosts (quantization, caching, and serving stack differ). `compat.openRouterRouting.only: [<provider slug>]` in the provider config makes every request carry `provider: {"only": [...]}`, so a stage runs on one serving stack; pick hosts from `GET https://openrouter.ai/api/v1/models/<id>/endpoints` (no key) and slugs from `GET https://openrouter.ai/api/v1/providers`.

### Arms

| Arm | archex surface | Stage |
| --- | --- | --- |
| A0 | none | pilot + confirmatory |
| H | annotation hook (omp `-e <hook module>`), archex not on `PATH` | pilot + confirmatory |
| HC | hook + CLI on `PATH` + CLI guide in the system prompt | pilot + confirmatory |
| M | archex MCP server in the profile's `mcp.json` (`archex mcp`, as `archex install-client omp` installs it), archex not on `PATH`, no hook; `--tools` adds `mcp__archex_context,mcp__archex_query_repo` | pilot + confirmatory |
| C | CLI on `PATH` + CLI guide, no hook | pilot only |

M needs the overlay's `mcp.startupTimeoutMs: 0`: omp registers MCP tools only after the server connects and checks `--tools` against what is registered, and the archex server starts slower than omp's default 250 ms window.

Muna honours effort for Qwen 3.8-27B (82 vs 271 reasoning tokens at low vs high) but not for Gemma 4-26B-A4B (3+3 requests, mean 1,033 vs 1,100, ratio 1.06, overlapping), so Gemma runs at `high` only. In G1 (2026-10-02/03) Muna answered every request for both models with `429 model_capacity_exhausted` (pre-registration, Stage 0 item).

`stealth/space-bunny-alpha` is an anonymous OpenRouter preview: $0 per token, `expiration_date: 2026-10-05` in OpenRouter's catalog, identity undisclosed (it may change behind the same id), and prompts and completions may be retained by the provider. It can exercise the route and the harness; it is not a pinned model for confirmatory data.

**Adding a campaign.** (1) Write `benchmarks/swe_ab/<provider>-models.yml` with one provider and its models; name the provider after omp's built-in host when omp special-cases it (omp 18.4.4 keys OpenRouter adaptations on the provider name `openrouter` as well as the URL, and the Stage 0 stub swaps only the URL) and pin `compat.thinkingFormat: openai` when the provider accepts `reasoning_effort`, so the stub-backed `effort_request_shape` check sees the real request shape. (2) Write `benchmarks/swe_ab/campaigns/<name>.yml` (`name`, `provider_config`, `configurations: [{label, model, thinking}]`). (3) Put the key on its own `NAME=value` line in the repo-root `.env`. (4) Run Stage 0 with `--campaign` (§4). A free model leaves the dollar ceiling inert: set `token_ceiling` in the plan (§7).

## 1. Host

SWE-bench Pro images are `linux/amd64`, and scoring runs each task's own test suite. Two hosts are supported; **a stage must run entirely on one kind**, because emulated wall times, timeouts, and (rarely) test outcomes differ. Every cell records `emulated: true|false`, and the validator refuses a stage that mixes them.

| | x86_64 Linux (campaign) | Apple silicon Mac (local, emulated) |
| --- | --- | --- |
| Runtime | Docker Engine 24+, daemon running | Docker Desktop with x86_64/amd64 emulation through Rosetta |
| CPU | ~32 vCPU, `--jobs` up to what the provider's capacity and rate limits tolerate (§6) | give Docker at most 80% of the cores; `--jobs 1` or `2` |
| Memory | 128 GB | at most ~60% of RAM to Docker (a 32 GB Mac: 20 GiB); agent + verifier run one container at a time per job |
| Disk | 1 TB | Docker disk image ≥ 150 GiB free (Pro images are multi-GB each); see §1.2 |
| Network | egress to `ghcr.io`, the campaign's provider host, PyPI/GitHub for setup | same |
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

Each Pro image is several GB and a stage touches dozens to hundreds. Cells run one model family at a time (§3.4), so a task's image is needed again for the next family. Pass `--prune-images` to the suite so it removes a task's image once that task's cells **for the current family** have run; docker re-pulls it when the next family reaches the task. Otherwise:

```bash
docker image rm ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>     # after that task's last cell of a family
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
# glibc's loader and the libraries Bun and omp's native addon link, for task images without glibc
# (the protonmail/webclients and some gravitational/teleport images are Alpine/musl), and the entry
# point that uses them only there:
docker run --rm --platform linux/amd64 -v "$WORK/omp-linux-x64:/opt/omp" oven/bun:1-debian sh -c \
  'mkdir -p /opt/omp/glibc && for f in /opt/omp/bin/bun /opt/omp/install/global/node_modules/@oh-my-pi/pi-natives-linux-x64/*.node; do ldd "$f"; done |
   awk "/=>/ {print \$3} /ld-linux/ {print \$1}" | sort -u | xargs -I{} cp -L {} /opt/omp/glibc/'
install -m 755 benchmarks/swe_ab/omp-entry.sh "$WORK/omp-linux-x64/bin/omp-entry"
# The entry point inside containers is then:
OMP_COMMAND=/opt/omp/bin/omp-entry

# The host's own omp, for Stage 0 and the no-spend rehearsals, must be the same build.
# `omp --version` must print 18.4.4 (a newer omp is refused by the cell schema); install it beside the default one:
mkdir -p "$WORK/omp-host" && BUN_INSTALL="$WORK/omp-host" bun install -g @oh-my-pi/pi-coding-agent@18.4.4
OMP_HOST="$WORK/omp-host/bin/omp"

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

### 2.2 The profile holds no credentials and no provider config

Provision the isolated profile once, with auto-update off, no memory, and no advisor, **without logging in to anything**:

```bash
omp --profile swebench            # configure only, then quit; do not run /login
PROFILE_DIR=~/.omp/profiles/swebench/agent
rm -f "$PROFILE_DIR"/agent.db "$PROFILE_DIR"/agent.db-*    # omp creates its SQLite store here; it is also omp's login vault
ls "$PROFILE_DIR"                 # config.yml, …; no agent.db, no extensions/, skills/, rules/, AGENTS.md
```

The harness copies the profile into each container per cell and then installs the frozen provider config over it as `models.yml` (§3.2): any `models.yml` the profile has is replaced, never merged. It **refuses** a profile that carries `agent.db*`, a `*.token` file, or an encrypted snapshot (`*.enc`): the suite exits before any cell, and the cell runner records a `harness_error` if it is reached anyway. The key never travels as a copied file.

## 3. The campaign's provider: credentials, provider config, capacity

### 3.1 The key

The key variable is the provider config's `apiKey` (`MUNA_ACCESS_KEY` for `campaigns/muna.yml`, `OPENROUTER_API_KEY` for `campaigns/openrouter-space-bunny.yml`). The suite takes it from its own environment, else from that variable's `NAME=` line of `--env-file` (default: the repository-root `.env`, gitignored; no other line is read). A docker run without the key exits before any cell. **Never print it**: nothing in the harness does, and a cell records only the variable *name* (`credential_env_names`).

Check it without spending anything: a chat request that names a model that does not exist generates nothing. A good key gets the request rejected for its model (Muna 404 `model_not_found`, OpenRouter 400 `not a valid model ID`, both measured 2026-10-03); a bad or missing key gets 401. Stage 0 runs exactly that request (`provider_key_accepted`, §4); by hand, with `BASE` the provider's `baseUrl` and `KEY_VAR` its key variable:

```bash
printf 'Authorization: Bearer %s\n' "${!KEY_VAR}" | curl -s -o /dev/null -w '%{http_code}\n' "$BASE/chat/completions" \
  -H @- -H 'Content-Type: application/json' \
  -d '{"model":"nobody/does-not-exist","max_tokens":1,"messages":[{"role":"user","content":"x"}]}'   # 400/404 = accepted, 401 = rejected
```

`GET <baseUrl>/models` needs no key at Muna or OpenRouter (it is what `provider_reachable_from_container` fetches).

### 3.2 The provider config and its prices

A provider config is an omp `models.yml` with one provider: `baseUrl`, `api: openai-completions`, `apiKey` (the name of the environment variable omp reads), and the models with `reasoning: true`, an effort `thinking` block, `contextWindow`, and `maxTokens: 65536` (every campaign so far). `compat: {thinkingFormat: openai}` makes omp send the effort as `reasoning_effort`: the Muna Qwen model needs it (without it omp sends `enable_thinking: true` and no effort for a model id containing `qwen`, so `--thinking low` and `--thinking high` would be byte-identical requests), and the OpenRouter config pins it so the request does not depend on omp's URL detection. Stage 0 `effort_request_shape` fails if a configuration's first request lacks the right `reasoning_effort`.

The file is frozen by its SHA-256 (cells record `provider_config_sha256`; the validator requires the campaign's current value, so one value per stage). It also holds the **prices**: `cost` is USD per million tokens (`input`, `output`, `cacheRead`, `cacheWrite`). Muna's are from its price list as given by the operator on 2026-10-02 (Qwen 3.8 27B $0.25 input / $0.02 cached / $1.75 output; Gemma 4 26B $0.065 / $0.02 / $0.30); OpenRouter's from its catalog (`GET https://openrouter.ai/api/v1/models`). `cacheWrite` is exempt (an `openai-completions` response reports no cache-write tokens). Changing a price changes the file's hash.

**Free models.** omp's cost model reads $0 for a model whose `input`, `output`, or `cacheRead` price is 0, so the dollar ceiling cannot stop a runaway. A docker run therefore refuses to start when any campaign model is unpriced and the plan sets no `token_ceiling` (§7); `validate` refuses such a directory too.

### 3.3 What the container receives, and what that exposes

An explicit allow-list: **one variable**, the campaign's key variable, and nothing else from the host environment (cloud credentials, other provider keys, and the rest are never forwarded). It is passed as a bare `-e NAME` with the value in the docker client's environment, so it does not appear in any process listing or argv. The validator refuses a cell whose recorded names are not exactly `[<campaign key variable>]`.

The agent runs with `--approval-mode yolo` and a `bash` tool, so **the agent can read the key from its own environment** and could spend credits with it or use the account. Use a key whose account holds only what the campaign may spend (OpenRouter keys carry a credit `limit`), and rotate it after every stage. Free previews may retain prompts and completions at the provider; SWE-bench Pro tasks are public, but the agent's environment and tool output go there too.

Egress is not restricted: the container needs the provider's host only, but this harness's default is Docker's `bridge` network with open egress, fixed at freeze and recorded per cell (`network`). A restrictive setup that is feasible but untested here: on Linux, `docker network create swe-ab`, `--network swe-ab`, and `iptables -I DOCKER-USER` rules for that subnet allowing DNS and the provider's host, dropping the rest. Docker Desktop has no host firewall for container traffic.

### 3.4 Capacity and scheduling

Some providers swap models in and out of shared GPU capacity: Muna's switches cost a wait (about 6 minutes was observed), during which requests get `429` with `model_loading` or `model_capacity_exhausted` and no `retry-after`. The suite therefore runs cells ordered by **(omp selector, task, configuration label, arm, repetition)**: every configuration of one selector, task by task, before the next model family. One family stays loaded for a whole pass; `--jobs N` bounds parallelism within it. Parallelism does not remove a swap wait, only the number of cells in flight. OpenRouter free models carry per-account request limits instead (§6).

## 4. Stage 0 checks (no model spend)

```bash
# Frozen Stage 1 draw (seed is a constant in the script) and the ids for the validity check
uv run python scripts/swe_ab_sample.py --tasks-root "$TASKS" --stage 1 --campaign "$CAMPAIGN" \
  --cost-ceiling <usd> [--token-ceiling <tokens>] \
  --plan-out stage1-plan.json --manifest-out stage1-sample.json --instances-out stage0-instances.txt

uv run python scripts/swe_ab_stage0.py --campaign "$CAMPAIGN" --output benchmarks/swe_ab/stage0.json --host \
  --tasks-root "$TASKS" --instances stage0-instances.txt \
  --omp-dir "$WORK/omp-linux-x64" --omp-command "$OMP_HOST" --container-omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary "$WORK/uv/uv"
jq '.summary, .gate, [.checks[] | select(.status != "pass") | {id, status, detail}]' benchmarks/swe_ab/stage0.json
```

`CAMPAIGN` is a campaign file (e.g. `benchmarks/swe_ab/campaigns/muna.yml`); the sampler requires `--token-ceiling` when the campaign has an unpriced model. `--omp-command` is the host's pinned omp, used by the local checks; `--container-omp-command` is the bundle's entry point inside a task container (`$OMP_COMMAND` of §2, also its default), used by `omp_runs_in_container`.

The draw is a pure function of the tasks root, the exclusions and the stage. An instance that fails Stage 0 validity (gold patch does not resolve, or the empty patch does not fail) goes into an exclusions file, `{"exclusions": [{"instance_id": ..., "reason": "gold_not_resolved" | "empty_not_failing", "source": ...}]}` (the `gold_empty_validity` check lists them in that shape under `exclusions`), and the sampler is re-run with `--exclusions <file> --force`: each failure is replaced by the next valid instance in its repository's rank order, and a repository that runs out has its shortfall reassigned to the one with the most left. Run Stage 0 on the replacements and repeat until the whole draw is valid, then keep `stage1-plan.json` and `stage1-sample.json`. Stage 2 uses the same exclusions plus `--stage1-plan stage1-plan.json` and `--stage2-tasks N` (§8), and refuses a plan that differs from its own Stage 1 recompute.

Without `--host` (or when `docker info` fails) the script runs only the local checks and leaves the container checks `requires_host`. The key check reads the campaign's key variable as in §3.1 (`--env-file` overrides the file):

| Check | What it establishes | Model call |
| --- | --- | --- |
| `omp_version` | the omp given by `--omp-command` is the pinned build (18.4.4) | no |
| `provider_route` | `omp models --json`, run against a temporary profile holding the campaign's provider config, lists every configuration's selector under the campaign's provider | no |
| `provider_key_accepted` | the zero-token request of §3.1 is rejected for its model (400/404/422; 401/403 fail; a missing key fails without any request; the key is never printed) | no (nothing is generated) |
| `frozen_identities` | SHA-256 of the CLI guide, the rendered hook module, the campaign file and its provider config, and `omp-campaign.yml` | no |
| `effort_request_shape` | per configuration, one stub-backed omp run (the stub serving that configuration's model id) whose first request body has `reasoning_effort` equal to the configuration's effort; fails when absent or different (the Qwen trap of §3.2) | no |
| `stub_rehearsal` … `validator_refuses_stub_cells` | one no-spend cell per arm (A0, H, HC, M, C) against the local stub (the committed dry-run plan, Muna campaign): tool list, system-prompt isolation, search-routing disclosure, annotation only in H/HC, no compressor output, validator refusal of stub-endpoint cells (the resolved base URL is not the campaign's) | no |
| `mcp_tools_in_m_arm` | the M cell advertises exactly the base tools plus `mcp__archex_context` and `mcp__archex_query_repo`, and no other arm advertises them | no |
| `mcp_call_executes_in_m_arm` | a stub-scripted M run calls `mcp__archex_query_repo` on the checkout and gets archex context back (the result size is recorded, never its text) | no |
| `gold_empty_validity` | per instance, the gold patch resolves and the empty patch fails (invalid instances leave the pool) | no |
| `omp_runs_in_container` | the pinned omp build starts inside each Pro image | no |
| `archex_indexes_in_container` | archex installs into `/opt/archex`, indexes the checkout to `fresh`, and the annotate entry is pre-warmed on a real search hit | no |
| `annotate_latency_in_container` | 10 annotate calls on 20 real hits of `git grep -w return` in the image's Python, Go, JS, and TS sources, after setup, timed end to end as the hook spawns them; fails when the median exceeds the campaign's hook budget, `ARCHEX_HOOK_TIMEOUT_SECONDS=5` (`HOOK_TIMEOUT_SECONDS`; the shipped default 0.5 s is far below emulated latency, about 1–2 s) (the hook would drop most annotations) | no |
| `emulated_wall_times` | container start, gold/empty scoring, omp start, and install+index seconds per instance; `emulated` and the container architecture | no |
| `provider_reachable_from_container` | a throwaway `alpine` container fetches `<baseUrl>/models` (unauthenticated) and sees the campaign's key variable and no other host variable | no |
| `bun_runs_under_emulation` | Bun's default x86-64 build starts under emulation, or the baseline build does (§2.1) | no |
| `one_real_cell_per_configuration` | §5 | **yes** |

The annotate **pre-warm** runs in every hook-arm cell's setup: `python -m archex.integrations.annotate_hook` is run once on a real `git grep` hit in the checkout, so the first call the agent sees is not the cold start (bytecode compilation, tokenizer load, cold caches) that could exceed the hook's 0.5 s budget and silently drop an annotation. It logs to `/dev/null`, must reach the search path (or the cell fails at setup), and its duration is recorded in `annotate_prewarm_seconds`.

Instances failing gold/empty validity leave the candidate pool before any agent run; list them in the pre-registration.

## 5. One real cell per configuration (spends credits — explicit go-ahead required)

This is the last Stage 0 check. **Do not run it until the prices are set (§3.2) and the maintainer says go.** One short task, arm `H` (it exercises the route, auth, usage reporting, and the annotation ledger), one repetition per configuration of the campaign. `cost_ceiling_usd` is **real money**: omp's usage × the prices in the campaign's provider config; set it to what you are willing to spend. A free campaign needs `token_ceiling` as well.

```json
{
  "name": "stage0-one-cell",
  "campaign": "benchmarks/swe_ab/campaigns/openrouter.yml",
  "tasks": [{"task_id": "<one short instance_id>", "repo": "<org/repo>"}],
  "models": ["qwen-3.8-27b@low", "qwen-3.8-27b@high", "gemma-4-26b-a4b-it@high"],
  "repetitions": {"H": 1},
  "cost_ceiling_usd": 5
}
```

```bash
uv run python scripts/run_swe_ab_suite.py run --plan stage0-one-cell.json --runtime docker \
  --output benchmarks/swe_ab/results/stage0 --work-root "$WORK/stage0" \
  --tasks-root "$TASKS" --omp-dir "$WORK/omp-linux-x64" --omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary "$WORK/uv/uv" \
  --prune-images --jobs 1
uv run python scripts/run_swe_ab_suite.py validate --plan stage0-one-cell.json \
  --input benchmarks/swe_ab/results/stage0
jq '{model, thinking, status, failure_reason, usage, hook: .hook_ledger, provider, provider_base_url, emulated, credential_env_names, quota}' \
  benchmarks/swe_ab/results/stage0/*/H/*.json
```

Pass: every cell `ok`, `usage` non-zero at every tier the provider reports, `provider` and `provider_base_url` are the campaign's, `credential_env_names` is `[<campaign key variable>]`, `usage.cost_usd` is non-zero for a priced campaign (the prices reached omp; a free campaign reads 0 by construction), `hook_ledger.annotated > 0` when the agent searched before editing. Record `quota` (omp retries, block phase) per configuration, and any first-request capacity wait.

## 6. Rate limits, capacity, and credits

Providers answer an unservable request with HTTP 429: Muna for capacity (`model_capacity_exhausted`, `model_loading`, no `retry-after`), OpenRouter for its account and model rate limits (free models carry daily and per-minute request limits; their size for a given model and key is not measured here). What omp does on a Muna capacity 429, measured against omp 18.4.4 with the local stub answering HTTP 429 with Muna's JSON body and no `retry-after` (the stub's `http_error` step with a `body`; no hosted call; the model `muna/@qwen/qwen-3.8-27b`, `--thinking high`, otherwise the frozen command line):

- **Default settings** (`retry.maxRetries` 10, `retry.maxDelayMs` 5 min): omp retries the request 10 times with its own backoff (500 ms doubling, capped at 8 s, jittered: 48–50 s of waiting in the stream's `auto_retry_start` events) and then fails with `429 No capacity for this model … (type=rate_limit_error param=model_capacity_exhausted)`. Each omp retry also spends about 30 s inside the provider client's own request retries, so the whole sequence took **398–400 s** wall-clock, before the first tool call and after one tool call, for `model_capacity_exhausted` and `model_loading` alike. That is shorter than the ~6-minute-plus swap waits Muna was seen to impose, so the default budget would end cells that only needed to wait.
- **Campaign overlay** `benchmarks/swe_ab/omp-campaign.yml` (omp's `--config`, identical in every arm and model, SHA-256 recorded per cell as `omp_config_sha256`): `retry.maxRetries: 20`, `retry.waitForUsageReset: false`. Re-measured with it: 20 retries, 118–122 s of omp backoff, **785–789 s** wall-clock before giving up (before the first tool call and after one, both error codes). `retry.baseDelayMs` and `retry.maxDelayMs` stay at their defaults; the 8 s backoff cap is not configurable in 18.4.4.
- **`retry.waitForUsageReset` stays false**: a usage-limit 429 that states a long reset still fails fast (`Provider requested 14400000ms wait, exceeds retry.maxDelayMs (300000ms)`; with the overlay, 0 retries, 2 s), so a cell is never held past its 60-minute cap.
- The 785–789 s are against a stub that answers instantly; a real provider's round-trip time adds to each request, so the figures are a floor.

### What the harness does

- **A cell that ends in a rate-limit or capacity block** (`429`, `model_loading`, `model_capacity_exhausted`, a final error from omp after its own retries, before its first tool call or mid-run) is **not a task outcome**. It is recorded with `failure_reason: quota_block` and `quota.block_phase` of `before_first_tool_call` or `mid_run`, its tokens counted in cost, and its artifact filed under `<output>/quota-blocked/` (never scored). There is no usage endpoint to poll, so the suite re-runs the cell after `--block-cooldown-seconds` (default 300), up to `--quota-retries` (default 2) per invocation; past that the blocked artifact stays as the cell's record, the next invocation files it and re-runs the cell, and `validate` refuses it meanwhile. omp's own in-run retries are recorded in `quota.omp_retries` and `quota.omp_retry_wait_seconds`. The identifiers `quota_block` and `quota-blocked/` name a rate-limit or capacity block, not a subscription quota.
- **Credit exhaustion** (HTTP 402, `insufficient credits|balance|funds`, `payment required`, `credits_required`, `out of credits`) is `failure_reason: credit_exhausted`. It is never retried and never scored: the attempt is filed under `quota-blocked/`, running cells finish, nothing new starts, and the suite exits with **status 5**. Top up the account, then run the same command again (resumable). OpenRouter documents 402 for insufficient credits; **Muna's credit-exhaustion response shape is unverified**: the pattern is the usual reading of such responses, and an exhaustion that matches neither pattern would be a recorded `provider_error` (retried once if before the first tool call), so watch the first one.
- **Provider failures that are neither** keep the earlier rule: a failure before the first tool call is retried once, logged (`retried: true`); anything later is a recorded `provider_error`.
- **Publication:** `validate` accepts a directory only when no cell at its cell path is a block (quota or credit), and reports the number of filed attempts (`quota_blocked`). Blocked attempts are disclosed in the analysis per arm.
- The ceilings (§7) are the runaway guard. Exit statuses: 3 cost or token ceiling, 5 credits exhausted.
- **A re-run starts clean.** Before each attempt the cell runner moves whatever an earlier attempt of the same cell left in its work directory into `<work dir>/prior-attempts/<n>/`, so an attempt reads only its own session, ledger, and patch, and the raw files of blocked attempts are kept.

Rehearsal, no spend (each outcome is classified by a real omp run against the stub; `$OMP_HOST` is the 18.4.4 build of §2):

```bash
cat > /tmp/quota-stub.json <<'EOF'
[{"tool": "grep", "args": {"i": "Locating the hash helper", "pattern": "hash_password"}},
 {"http_error": 429, "message": "You have hit your usage limit. Your limit will reset in 4 hours.", "retry_after": 14400}]
EOF
cat > /tmp/credit-stub.json <<'EOF'
[{"tool": "grep", "args": {"i": "Locating the hash helper", "pattern": "hash_password"}},
 {"http_error": 402, "message": "insufficient credits: top up your balance"}]
EOF
for kind in quota credit; do
  uv run python scripts/run_swe_ab_suite.py run --plan benchmarks/swe_ab/dry-run-plan.json \
    --runtime local --output /tmp/swe-ab/$kind-cells --work-root /tmp/swe-ab/$kind-work \
    --omp-command "$OMP_HOST" --stub-script /tmp/$kind-stub.json \
    --block-cooldown-seconds 1 --quota-retries 0
done
# quota: every arm's cell is recorded as quota_block, phase mid_run (the 4-hour reset fails fast, no retry loop)
# credit: the first cell is filed under quota-blocked/ as credit_exhausted and the run exits 5
```

A Muna-shaped capacity body (`{"http_error": 429, "body": {"error": {"code": "model_capacity_exhausted", ...}}}`, no `retry_after`) drives omp's full retry loop instead: about 13 minutes per cell with the overlay.

## 7. Cost and token ceilings

- The plan's `cost_ceiling_usd` is the hard cumulative ceiling for the stage, in **real money**. `--cost-ceiling` can only lower it.
- The plan's optional `token_ceiling` is a hard cumulative ceiling on billed tokens (input + output + cache read + cache write, summed over every request of every cell and filed attempt). It is **required when any campaign model is unpriced** (a free preview), because omp's cost model reads $0 there; `--token-ceiling` can only lower it.
- The suite checks, before starting each cell, that spent cost (and tokens) plus the largest recorded cell's cost (and tokens) for every cell still running stays under each ceiling; otherwise it waits for running cells, records them, and exits with status 3.
- Recorded cost is omp's per-request `usage.cost.total`: the usage the provider reports × the prices in the campaign's provider config, summed per cell, **including blocked attempts**. Resumed runs count existing cells' cost and tokens toward the ceilings.
- `--jobs N` runs N cells in parallel; keep N within what the provider's capacity and rate limits tolerate (§3.4, §6) and the host's resources (§1).

## 8. A0 feasibility check, Stage 1, and Stage 2

**A0 feasibility check (before Stage 1; never data).** Before spending on the pilot, run the first 6 tasks of the Stage 1 plan × the campaign's configurations × A0 × 1 repetition (18 cells for the Muna campaign), to decide whether SWE-bench Pro is solvable enough by these models to proceed. Its results stay **outside** `benchmarks/swe_ab/results/`, are never analysed or pooled, and are used only for that decision. If every configuration resolves 0 of 6, stop and revisit the benchmark (pre-registration, kill criterion 3).

```bash
jq '{name: "a0-feasibility", campaign: .campaign, tasks: .tasks[:6], models: .models,
     repetitions: {A0: 1}, cost_ceiling_usd: <usd>} + (if .token_ceiling then {token_ceiling: <tokens>} else {} end)' \
  stage1-plan.json > "$WORK/a0-feasibility-plan.json"
uv run python scripts/run_swe_ab_suite.py run --plan "$WORK/a0-feasibility-plan.json" --runtime docker \
  --output "$WORK/a0-feasibility" --work-root "$WORK/a0-feasibility-work" \
  --tasks-root "$TASKS" --omp-dir "$WORK/omp-linux-x64" --omp-command "$OMP_COMMAND" \
  --profile-dir "$PROFILE_DIR" --archex-wheel "$ARCHEX_WHEEL" --uv-binary "$WORK/uv/uv" --prune-images --jobs 2
jq -s 'group_by(.model) | map({model: .[0].model, resolved: (map(select(.resolved)) | length), cells: length})' \
  "$WORK"/a0-feasibility/*/A0/*.json
```

**Stage 1** (the pilot) runs on the draft pre-registration with its own ceiling; freeze the pre-registration after it (fill every *(set at freeze)* field, including the two config SHA-256 values) and before any Stage 2 cell. Run each stage with its plan exactly as in §5 with `--runtime docker`, and validate the result directory before any analysis. The run resumes by skipping cells whose artifact exists. A harness defect found after data exists means discarding and restarting the affected stage, never repairing cells. Rotate the key between stages (§3.3).

The pre-registered analysis (`scripts/swe_ab_analysis.py`) refuses anything `validate` refuses:

```bash
uv run python scripts/swe_ab_analysis.py --plan stage1-plan.json \
  --input benchmarks/swe_ab/results/stage1 --output benchmarks/evidence/r3x-swe-ab-pilot.json
# Stage 1 gates: .gates.solve_floor, .gates.headroom, .gates.hc_adoption, .gates.mcp_adoption,
# .gates.hook_activity, .gates.a0_noise, .gates.stage2_size
```

**Stage 2 size.** The non-inferiority guardrail needs about 370 pairs per arm; the two-sided completion test (H5) at 80% power for a 5-point difference, Holm over the m primary arms kept after the adoption gates, needs n(3) = 626, n(2) = 568, n(1) = 469 pairs. Stage 2 tasks = max(100, ceil(max(370, n(m)) / k)), where k is the number of configurations whose Stage 1 A0 solve rate is at least 15% (`gates.solve_floor`); k = 3 and m = 3 give 209 (the report's `gates.stage2_size` states it). Draw it with `--stage2-tasks`, which the sampler requires for Stage 2 (at least 100) and records in the manifest, adding `--without-hc` / `--without-m` for an arm an adoption gate dropped; Stage 2 keeps the Stage 1 plan's campaign:

```bash
uv run python scripts/swe_ab_sample.py --tasks-root "$TASKS" --stage 2 --stage2-tasks <N> --campaign "$CAMPAIGN" \
  --exclusions exclusions.json --stage1-plan stage1-plan.json --cost-ceiling <usd> [--token-ceiling <tokens>] \
  --plan-out stage2-plan.json --manifest-out stage2-sample.json --instances-out stage2-instances.txt
uv run python scripts/swe_ab_analysis.py --plan stage2-plan.json \
  --input benchmarks/swe_ab/results/stage2 --output benchmarks/evidence/r3x-swe-ab.json \
  --pilot-analysis benchmarks/evidence/r3x-swe-ab-pilot.json
```

With `--pilot-analysis`, Stage 2's pooled guardrail excludes configurations whose pilot floor failed (they are reported per configuration, labelled `uninformative: below the 15% solve-rate floor`) and reports `not_estimable` if none pass; efficiency stays per configuration regardless.

## 9. No-spend rehearsal (any machine with omp 18.4.4 and archex)

```bash
uv run python scripts/run_swe_ab_suite.py run --plan benchmarks/swe_ab/dry-run-plan.json \
  --runtime local --output /tmp/swe-ab/cells --work-root /tmp/swe-ab/work \
  --omp-command "$OMP_HOST" --stub-script benchmarks/swe_ab/stub-script.json
uv run python scripts/run_swe_ab_suite.py validate --plan benchmarks/swe_ab/dry-run-plan.json \
  --input /tmp/swe-ab/cells     # exits 1: the cells' resolved base URL is the stub's, not the campaign's
uv run python scripts/swe_ab_analysis.py --plan benchmarks/swe_ab/dry-run-plan.json \
  --input /tmp/swe-ab/cells --output /tmp/swe-ab/analysis.json     # refuses for the same reason
```

The stub (`scripts/swe_ab_stub_provider.py`) replays a `grep`, a `read`, an `edit`, and a stop over the OpenAI chat-completions SSE protocol; each cell runs with an isolated `HOME` and its own copy of the fixture repository. The rehearsal profile is the plan's campaign provider config (the dry-run plan names `campaigns/muna.yml`) with `baseUrl` pointed at the stub (same provider name, model ids, and thinking settings, the key dropped), so omp builds the requests a campaign cell would. The local runtime carries no credentials (the campaign's key variable in the environment is stripped from the agent) and records `emulated: false`, `credential_env_names: []`.
