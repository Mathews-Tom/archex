#!/usr/bin/env bash
set -Euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

output_dir_rel=".archex/e2e-tokens"
baseline_dir_rel=".archex/e2e-tier2"
log_file_rel=".docs/pipeline.log"

output_dir="$repo_root/$output_dir_rel"
baseline_dir="$repo_root/$baseline_dir_rel"
log_file="$repo_root/$log_file_rel"


format_duration() {
    local duration="$1"

    printf '%02dh:%02dm:%02ds' \
        "$((duration / 3600))" \
        "$(((duration % 3600) / 60))" \
        "$((duration % 60))"
}

prepare_run_artifacts() {
    mkdir -p "$repo_root/.archex" "$repo_root/.docs"
    rm -rf "$output_dir" "$log_file"
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

run_pipeline() {
    local status=0

    run_step "Benchmark Run" \
        uv run archex benchmark run \
        --query-fusion \
        --rerank \
        --embedder jina-v2 \
        --tasks-dir benchmarks/tasks \
        --output "$output_dir_rel" || status=$?

    run_step "Benchmark Gate" \
        uv run archex benchmark gate \
        --input "$output_dir_rel" \
        --baseline "$baseline_dir_rel" \
        --warn-latency-ms 3000 || status=$?

    run_step "Dogfood Delta" \
        uv run archex dogfood . \
        --all \
        --baseline benchmarks/dogfood_baseline.json \
        --format dogfood-delta || status=$?

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
        echo "Output directory: $output_dir_rel"
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
