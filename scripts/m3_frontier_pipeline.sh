#!/usr/bin/env bash
# Local-operator M3 external quality frontier pipeline: runs the default,
# fast, balanced, and (optionally) symbolic-rerank candidate lanes in one
# "frontier" evidence run, the cAST lane in a second run (cAST needs its own
# --chunker cast invocation of the archex_query strategy, which cannot share
# a report with the default archex_query result -- see
# src/archex/benchmark/external_frontier.py), then emits M3
# scorecards, runs the multidimensional same-run promotion gate for the
# fast/balanced/symbolic-rerank candidates against their archex_query
# control, and runs an absolute-threshold gate on the cAST directory.
#
# This is local-operator only: it executes the full task corpus (network
# clones for every pinned external task) and is never invoked from CI.
set -Euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

resolve_repo_path() {
    local path="$1"

    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$repo_root" "$path" ;;
    esac
}

is_enabled() {
    case "$1" in
        1 | true | TRUE | yes | YES | on | ON) return 0 ;;
        0 | false | FALSE | no | NO | off | OFF) return 1 ;;
        *)
            echo "Invalid boolean value: $1" >&2
            exit 2
            ;;
    esac
}

append_option_if_set() {
    local array_name="$1"
    local option="$2"
    local value="${3:-}"
    local option_q value_q

    if [[ -n "$value" ]]; then
        printf -v option_q '%q' "$option"
        printf -v value_q '%q' "$value"
        eval "$array_name+=($option_q $value_q)"
    fi
}

format_duration() {
    local duration="$1"

    printf '%02dh:%02dm:%02ds' \
        "$((duration / 3600))" \
        "$(((duration % 3600) / 60))" \
        "$((duration % 60))"
}

run_step() {
    local label="$1"
    shift

    local cmd="$*"
    local start end duration status

    echo
    printf -- '=+%.0s' {1..25}
    echo
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') : Starting \"$label\" ====="
    echo "Command: $cmd"

    start=$(date +%s)
    if "$@"; then
        status=0
    else
        status=$?
    fi
    end=$(date +%s)
    duration=$((end - start))

    if [[ "$status" -eq 0 ]]; then
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') : Completed \"$label\" ====="
    else
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') : FAILED \"$label\" (exit $status) ====="
    fi
    echo "===== Time taken: $(format_duration "$duration") ====="
    printf -- '=+%.0s' {1..25}
    echo
    echo
    return "$status"
}

# Corpus and output settings. Point ARCHEX_M3_TASKS_DIR at
# benchmarks/sealed_tasks plus ARCHEX_M3_ALLOW_SEALED_CORPUS=1 to run the
# sealed holdout instead of the public bounded corpus.
tasks_dir_rel="${ARCHEX_M3_TASKS_DIR:-benchmarks/tasks}"
output_root_rel="${ARCHEX_M3_OUTPUT_ROOT:-.archex/m3-frontier}"
allow_sealed_corpus="${ARCHEX_M3_ALLOW_SEALED_CORPUS:-0}"
include_symbolic_rerank="${ARCHEX_M3_INCLUDE_SYMBOLIC_RERANK:-0}"
self_only="${ARCHEX_M3_SELF_ONLY:-0}"

# Promotion-gate thresholds for the same-run candidates (fast/balanced/
# symbolic-rerank vs the archex_query control).
min_token_efficiency_with_completion="${ARCHEX_M3_MIN_TOKEN_EFFICIENCY_WITH_COMPLETION:-0.08}"
max_p95_warm_latency_ms="${ARCHEX_M3_MAX_P95_WARM_LATENCY_MS:-5000}"

# Absolute-threshold gate for the cAST directory (see run_pipeline's
# run_cast_gate note on why a --baseline regression gate isn't reachable
# here today).
cast_min_recall="${ARCHEX_M3_CAST_MIN_RECALL:-0.60}"
cast_min_f1="${ARCHEX_M3_CAST_MIN_F1:-0.30}"
cast_min_mrr="${ARCHEX_M3_CAST_MIN_MRR:-0.55}"

log_file_rel="${ARCHEX_M3_LOG_FILE:-logs/m3_frontier_pipeline.log}"

run_frontier="${ARCHEX_M3_RUN_FRONTIER:-1}"
run_cast="${ARCHEX_M3_RUN_CAST:-1}"
run_scorecards="${ARCHEX_M3_RUN_SCORECARDS:-1}"
run_promotion_gates="${ARCHEX_M3_RUN_PROMOTION_GATES:-1}"
run_cast_gate="${ARCHEX_M3_RUN_CAST_GATE:-1}"

output_root="$(resolve_repo_path "$output_root_rel")"
frontier_dir="$output_root/frontier"
cast_dir="$output_root/cast"
scorecards_dir="$output_root/scorecards"
log_file="$(resolve_repo_path "$log_file_rel")"

frontier_strategies=(archex_query archex_query_profile_fast archex_query_profile_balanced)
if is_enabled "$include_symbolic_rerank"; then
    frontier_strategies+=(archex_query_symbolic_rerank)
fi

prepare_run_artifacts() {
    mkdir -p "$(dirname -- "$log_file")"
    rm -f "$log_file"
    if is_enabled "$run_frontier"; then
        rm -rf "$frontier_dir"
    fi
    if is_enabled "$run_cast"; then
        rm -rf "$cast_dir"
    fi
    if is_enabled "$run_scorecards"; then
        rm -rf "$scorecards_dir"
        mkdir -p "$scorecards_dir"
    fi
}

run_scorecard() {
    local input_dir="$1"
    local strategy="$2"
    local label="$3"
    local -a cmd=(
        uv run archex benchmark scorecard
        --input "$input_dir"
        --tasks-dir "$tasks_dir_rel"
        --strategy "$strategy"
        --format markdown
        --output "$scorecards_dir/scorecard-$label.json"
    )
    if is_enabled "$allow_sealed_corpus"; then
        cmd+=(--allow-sealed-corpus)
    fi
    run_step "Scorecard: $label" "${cmd[@]}"
}

run_promotion_gate() {
    local candidate="$1"
    local -a cmd=(
        uv run archex benchmark gate
        --input "$frontier_dir"
        --tasks-dir "$tasks_dir_rel"
        --promotion-strategy "$candidate"
        --control-strategy archex_query
        --min-token-efficiency-with-completion "$min_token_efficiency_with_completion"
        --max-p95-warm-latency-ms "$max_p95_warm_latency_ms"
    )
    run_step "Promotion Gate: $candidate vs archex_query" "${cmd[@]}"
}

run_pipeline() {
    local status=0
    local -a frontier_run_cmd cast_run_cmd cast_gate_cmd
    local -a run_flags=()
    local symbolic_label=""

    run_flags+=(--tasks-dir "$tasks_dir_rel")
    if is_enabled "$self_only"; then
        run_flags+=(--self-only)
    fi
    if is_enabled "$allow_sealed_corpus"; then
        run_flags+=(--allow-sealed-corpus)
    fi
    if is_enabled "$include_symbolic_rerank"; then
        symbolic_label=", symbolic-rerank"
    fi

    if is_enabled "$run_frontier"; then
        frontier_run_cmd=(
            uv run archex benchmark run "${run_flags[@]}"
            --output "$frontier_dir" --chunker default --no-progress
            # NOTE: --warm-cache was tried here to satisfy
            # --max-p95-warm-latency-ms's cache_state=="warm" requirement,
            # but it produced an incomplete per-task strategy set for the
            # balanced (module_prefilter) profile on at least one task in
            # this corpus/environment (a pre-existing runner interaction,
            # not introduced by M3). Until that is root-caused, expect
            # promotion-gate warm_latency_unmeasured violations here; treat
            # them as a real, documented NO-GO reason rather than suppress
            # them by re-adding --warm-cache blindly.
        )
        for strategy in "${frontier_strategies[@]}"; do
            frontier_run_cmd+=(--strategy "$strategy")
        done
        run_step "Frontier Run (default, fast, balanced${symbolic_label})" \
            "${frontier_run_cmd[@]}" || status=$?
    fi

    if is_enabled "$run_cast"; then
        cast_run_cmd=(
            uv run archex benchmark run "${run_flags[@]}"
            --output "$cast_dir" --chunker cast --strategy archex_query --no-progress
        )
        run_step "cAST Run" "${cast_run_cmd[@]}" || status=$?
    fi

    if is_enabled "$run_scorecards"; then
        run_scorecard "$frontier_dir" archex_query default || status=$?
        run_scorecard "$frontier_dir" archex_query_profile_fast fast || status=$?
        run_scorecard "$frontier_dir" archex_query_profile_balanced balanced || status=$?
        if is_enabled "$include_symbolic_rerank"; then
            run_scorecard "$frontier_dir" archex_query_symbolic_rerank symbolic-rerank || status=$?
        fi
        run_scorecard "$cast_dir" archex_query cast || status=$?
    fi

    if is_enabled "$run_promotion_gates"; then
        run_promotion_gate archex_query_profile_fast || status=$?
        run_promotion_gate archex_query_profile_balanced || status=$?
        if is_enabled "$include_symbolic_rerank"; then
            run_promotion_gate archex_query_symbolic_rerank || status=$?
        fi
    fi

    if is_enabled "$run_cast_gate"; then
        # `archex benchmark gate --baseline` requires the two evidence
        # directories' retrieval_options to match exactly
        # (validate_baseline_coverage), and `chunker` is part of that
        # options object -- so a true default-vs-cast regression gate via
        # --baseline is not currently reachable for two different-chunker
        # directories (a pre-existing evidence.py constraint, not
        # introduced by M3). This runs the absolute-threshold gate on the
        # cAST directory instead: it proves cAST clears the same floors
        # independently. Compare scorecards/scorecard-default.json against
        # scorecards/scorecard-cast.json for the actual recall/F1/MRR delta
        # -- both already carry per-task_id, per-strategy aggregates.
        cast_gate_cmd=(
            uv run archex benchmark gate
            --input "$cast_dir" --tasks-dir "$tasks_dir_rel"
            --min-recall "$cast_min_recall" --min-f1 "$cast_min_f1" --min-mrr "$cast_min_mrr"
        )
        run_step "cAST Absolute-Threshold Gate" "${cast_gate_cmd[@]}" || status=$?
    fi

    return "$status"
}

run_foreground() {
    prepare_run_artifacts
    cd "$repo_root"

    (
        local pipeline_start pipeline_end pipeline_status total_duration

        pipeline_start=$(date +%s)
        pipeline_status=0

        echo "=================================================="
        echo "M3 frontier pipeline started at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Repository root: $repo_root"
        echo "Tasks directory: $tasks_dir_rel"
        echo "Output root: $output_root_rel"
        echo "Symbolic-rerank lane included: $include_symbolic_rerank"
        echo "Log file: $log_file_rel"
        echo "=================================================="

        run_pipeline || pipeline_status=$?

        pipeline_end=$(date +%s)
        total_duration=$((pipeline_end - pipeline_start))

        echo
        echo "=================================================="
        echo "M3 frontier pipeline completed at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Total time taken: $(format_duration "$total_duration")"
        echo "Pipeline exit status: $pipeline_status"
        echo "=================================================="
        exit "$pipeline_status"
    ) 2>&1 | tee "$log_file"
}

main() {
    if (($# > 0)); then
        echo "m3_frontier_pipeline.sh does not accept arguments; configure it via ARCHEX_M3_* environment variables (see the script header)." >&2
        return 2
    fi

    run_foreground
}

main "$@"
