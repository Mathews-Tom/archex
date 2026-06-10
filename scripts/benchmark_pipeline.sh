#!/usr/bin/env bash
set -Euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_path() {
    local path="$1"

    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$repo_root" "$path" ;;
    esac
}


repo_root="$(cd -- "$script_dir/.." && pwd)"

# Architecture-quality smoke settings. These defaults are short and local; set
# ARCHEX_ARCH_BASELINE_DIR only after an operator has accepted baseline results.

arch_tasks_dir_rel="${ARCHEX_ARCH_TASKS_DIR:-benchmarks/arch_tasks}"
arch_output_dir_rel="${ARCHEX_ARCH_OUTPUT_DIR:-.archex/arch-quality-current}"
arch_baseline_dir_rel="${ARCHEX_ARCH_BASELINE_DIR:-}"
arch_min_boundary_f1="${ARCHEX_ARCH_MIN_BOUNDARY_F1:-0.80}"
arch_min_pattern_precision="${ARCHEX_ARCH_MIN_PATTERN_PRECISION:-0.80}"
arch_min_pattern_recall="${ARCHEX_ARCH_MIN_PATTERN_RECALL:-0.80}"
arch_min_interface_completeness="${ARCHEX_ARCH_MIN_INTERFACE_COMPLETENESS:-0.80}"

# Retrieval benchmark settings. Baseline and latency gates stay opt-in so the
# same script works for first runs, local experiments, and release checks.
benchmark_tasks_dir_rel="${ARCHEX_BENCHMARK_TASKS_DIR:-benchmarks/tasks}"
benchmark_output_dir_rel="${ARCHEX_BENCHMARK_OUTPUT_DIR:-.archex/benchmark-current}"
benchmark_baseline_dir_rel="${ARCHEX_BENCHMARK_BASELINE_DIR:-}"
benchmark_warn_latency_ms="${ARCHEX_BENCHMARK_WARN_LATENCY_MS:-}"

# Dogfood settings. The default baseline is the checked-in accepted dogfood
# baseline; ARCHEX_DOGFOOD_ARGS can add command-specific flags.
dogfood_source="${ARCHEX_DOGFOOD_SOURCE:-.}"
dogfood_baseline_rel="${ARCHEX_DOGFOOD_BASELINE:-benchmarks/dogfood_baseline.json}"
dogfood_format="${ARCHEX_DOGFOOD_FORMAT:-text}"

# Log under logs/ rather than documentation; callers can override this path.
log_file_rel="${ARCHEX_BENCHMARK_LOG_FILE:-logs/benchmark_pipeline.log}"

# Stage toggles accept 1/0, true/false, yes/no, or on/off.
run_arch_benchmark="${ARCHEX_RUN_ARCH_BENCHMARK:-1}"
run_retrieval_benchmark="${ARCHEX_RUN_RETRIEVAL_BENCHMARK:-1}"
run_dogfood="${ARCHEX_RUN_DOGFOOD:-1}"

arch_output_dir="$(resolve_repo_path "$arch_output_dir_rel")"
benchmark_output_dir="$(resolve_repo_path "$benchmark_output_dir_rel")"
log_file="$(resolve_repo_path "$log_file_rel")"


format_duration() {
    local duration="$1"

    printf '%02dh:%02dm:%02ds' \
        "$((duration / 3600))" \
        "$(((duration % 3600) / 60))" \
        "$((duration % 60))"
}

prepare_run_artifacts() {
    # Clear only artifacts this script owns so each invocation starts clean
    # without touching accepted baselines or unrelated local cache files.
    mkdir -p \
        "$(dirname -- "$log_file")" \
        "$(dirname -- "$arch_output_dir")" \
        "$(dirname -- "$benchmark_output_dir")"
    rm -rf "$arch_output_dir" "$benchmark_output_dir" "$log_file"
}

run_step() {
    local label="$1"
    shift

    local cmd="$*"
    local start end duration status

    if [[ -z "$label" ]]; then
        label="$cmd"
    fi

    echo
    printf -- '=+%.0s' {1..25}; echo
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
    printf -- '=+%.0s' {1..25}; echo
    echo
    return "$status"
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





run_pipeline() {
    local status=0
    local -a arch_gate_cmd arch_report_cmd
    local -a benchmark_gate_cmd benchmark_report_cmd benchmark_run_cmd
    local -a dogfood_cmd
    local -a benchmark_gate_extra_args benchmark_run_extra_args dogfood_extra_args

    # Keep the architecture stage independent from retrieval/dogfood so it can
    # run as a quick smoke or as the first stage of the full local pipeline.
    if is_enabled "$run_arch_benchmark"; then
        run_step "Architecture Benchmark Run" \
            uv run archex benchmark arch run \
            --tasks-dir "$arch_tasks_dir_rel" \
            --output "$arch_output_dir_rel" || status=$?

        arch_report_cmd=(
            uv run archex benchmark arch report
            --input "$arch_output_dir_rel"
        )
        if [[ -n "$arch_baseline_dir_rel" ]]; then
            arch_report_cmd+=(--baseline "$arch_baseline_dir_rel")
        fi
        run_step "Architecture Benchmark Report" "${arch_report_cmd[@]}" || status=$?

        arch_gate_cmd=(
            uv run archex benchmark arch gate
            --input "$arch_output_dir_rel"
            --min-boundary-f1 "$arch_min_boundary_f1"
            --min-pattern-precision "$arch_min_pattern_precision"
            --min-pattern-recall "$arch_min_pattern_recall"
            --min-interface-completeness "$arch_min_interface_completeness"
        )
        if [[ -n "$arch_baseline_dir_rel" ]]; then
            arch_gate_cmd+=(--baseline "$arch_baseline_dir_rel")
        fi
        run_step "Architecture Benchmark Gate" "${arch_gate_cmd[@]}" || status=$?
    fi

    # Retrieval defaults are intentionally plain. Use ARCHEX_BENCHMARK_RUN_ARGS
    # for opt-in strategies such as query fusion, rerankers, or custom embedders.
    if is_enabled "$run_retrieval_benchmark"; then
        benchmark_run_cmd=(
            uv run archex benchmark run
            --tasks-dir "$benchmark_tasks_dir_rel"
            --output "$benchmark_output_dir_rel"
            --no-progress
        )
        if [[ -n "${ARCHEX_BENCHMARK_RUN_ARGS:-}" ]]; then
            IFS=' ' read -r -a benchmark_run_extra_args <<< "$ARCHEX_BENCHMARK_RUN_ARGS"
            benchmark_run_cmd+=("${benchmark_run_extra_args[@]}")
        fi
        run_step "Benchmark Run" "${benchmark_run_cmd[@]}" || status=$?

        benchmark_report_cmd=(
            uv run archex benchmark report
            --input "$benchmark_output_dir_rel"
            --format markdown
        )
        run_step "Benchmark Report" "${benchmark_report_cmd[@]}" || status=$?

        benchmark_gate_cmd=(
            uv run archex benchmark gate
            --input "$benchmark_output_dir_rel"
        )
        if [[ -n "$benchmark_baseline_dir_rel" ]]; then
            benchmark_gate_cmd+=(--baseline "$benchmark_baseline_dir_rel")
        fi
        if [[ -n "$benchmark_warn_latency_ms" ]]; then
            benchmark_gate_cmd+=(--warn-latency-ms "$benchmark_warn_latency_ms")
        fi
        if [[ -n "${ARCHEX_BENCHMARK_GATE_ARGS:-}" ]]; then
            IFS=' ' read -r -a benchmark_gate_extra_args <<< "$ARCHEX_BENCHMARK_GATE_ARGS"
            benchmark_gate_cmd+=("${benchmark_gate_extra_args[@]}")
        fi
        run_step "Benchmark Gate" "${benchmark_gate_cmd[@]}" || status=$?
    fi

    # Dogfood runs last and still executes after prior failures so one run
    # reports every failing stage before returning a non-zero status.
    if is_enabled "$run_dogfood"; then
        dogfood_cmd=(
            uv run archex dogfood "$dogfood_source"
            --all
            --baseline "$dogfood_baseline_rel"
            --format "$dogfood_format"
        )
        if [[ -n "${ARCHEX_DOGFOOD_ARGS:-}" ]]; then
            IFS=' ' read -r -a dogfood_extra_args <<< "$ARCHEX_DOGFOOD_ARGS"
            dogfood_cmd+=("${dogfood_extra_args[@]}")
        fi
        run_step "Dogfood" "${dogfood_cmd[@]}" || status=$?
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
        echo "Pipeline started at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Repository root: $repo_root"
        echo "Architecture output directory: $arch_output_dir_rel"
        echo "Benchmark output directory: $benchmark_output_dir_rel"
        echo "Log file: $log_file_rel"
        echo "=================================================="

        run_pipeline || pipeline_status=$?

        pipeline_end=$(date +%s)
        total_duration=$((pipeline_end - pipeline_start))

        echo
        echo "=================================================="
        echo "Pipeline completed at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Total time taken: $(format_duration "$total_duration")"
        echo "Pipeline exit status: $pipeline_status"
        echo "=================================================="
        exit "$pipeline_status"
    ) 2>&1 | tee "$log_file"
}

main() {
    if (($# > 0)); then
        echo "benchmark_pipeline.sh does not accept arguments; run it as: bash scripts/benchmark_pipeline.sh" >&2
        return 2
    fi

    run_foreground
}

main "$@"
