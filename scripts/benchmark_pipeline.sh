#!/usr/bin/env bash
set -Euo pipefail

output_dir=".archex/e2e-tokens"
log_file=".docs/pipeline.log"

mkdir -p .archex .docs
rm -rf "$output_dir" "$log_file"

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

    if [[ -z "$label" ]]; then
        label="$cmd"
    fi

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
        --output "$output_dir" || status=$?

    run_step "Benchmark Gate" \
        uv run archex benchmark gate \
        --input "$output_dir" \
        --warn-latency-ms 3000 || status=$?

    run_step "Dogfood Delta" \
        uv run archex dogfood . \
        --all \
        --baseline benchmarks/dogfood_baseline.json \
        --format dogfood-delta || status=$?

    return "$status"
}

{
    pipeline_start=$(date +%s)
    pipeline_status=0

    echo "=================================================="
    echo "Pipeline started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Output directory: $output_dir"
    echo "Log file: $log_file"
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
} 2>&1 | tee "$log_file"
