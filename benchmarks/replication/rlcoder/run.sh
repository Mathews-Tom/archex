#!/usr/bin/env bash
# S0 replication run: RLCoder Table II, RepoEval line-level, DeepSeekCoder-1B.
#
# Two arms over the identical 1600-task split, differing only in the retrieval
# model. Everything else -- generator, weights, context budgets, decoding, and
# metric -- is held fixed. Run after `prepare.py`, and only after the
# pre-registration `.docs/spikes/S0-replication-gate.md` has merged.
#
# Usage: run.sh <work-dir> <output-dir>
set -euo pipefail

WORK_DIR="${1:?usage: run.sh <work-dir> <output-dir>}"
OUT_DIR="${2:?usage: run.sh <work-dir> <output-dir>}"
HARNESS="${WORK_DIR}/RLCoder"

if [[ ! -f "${WORK_DIR}/pins.json" ]]; then
  echo "no pins.json in ${WORK_DIR}; run prepare.py first" >&2
  exit 1
fi

cd "${HARNESS}"
ln -sfn "${WORK_DIR}/data" data

export PYTHONUNBUFFERED=1
export RLCODER_DEVICE="${RLCODER_DEVICE:-}"

# Record the exact invocation, including the device override, so analyze.py can
# put a runnable command in the evidence artifact instead of a template.
mkdir -p "${OUT_DIR}"
printf 'RLCODER_DEVICE=%s %s %s\n' "${RLCODER_DEVICE:-auto}" "$0" "$*" > "${OUT_DIR}/command.txt"

COMMON=(
  --eval --enable_generation --weighted_keywords
  --eval_datasets repoeval_line
  --generator_model_path "${WORK_DIR}/models/deepseek-coder-1.3b-base"
  --generator_max_crossfile_length 1536
  --generator_max_context_length 2048
  --generator_batch_size_per_gpu 8
  --retriever_batch_size_per_gpu 32
  --num_workers 4
)

echo "=== arm rawrag start $(date -u +%FT%TZ) ==="
python main.py "${COMMON[@]}" \
  --inference_type unixcoder \
  --retriever_model_path "${WORK_DIR}/models/unixcoder-base" \
  --output_dir "${OUT_DIR}/rawrag"
echo "=== arm rawrag done $(date -u +%FT%TZ) ==="

echo "=== arm rlcoder start $(date -u +%FT%TZ) ==="
python main.py "${COMMON[@]}" \
  --inference_type unixcoder_with_rl \
  --retriever_model_path "${WORK_DIR}/models/RLRetriever" \
  --output_dir "${OUT_DIR}/rlcoder"
echo "=== arm rlcoder done $(date -u +%FT%TZ) ==="
