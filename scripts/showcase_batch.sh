#!/usr/bin/env bash
set -euo pipefail

LOGFILE="showcase_results.log"
BUDGET=8192

REPOS=(
  "https://github.com/pallets/flask"          # Python — web framework, ~80 files
  "https://github.com/psf/requests"           # Python — HTTP library, ~60 files
  "https://github.com/fastapi/fastapi"        # Python — async web framework, ~200 files
  "https://github.com/django/django"          # Python — large web framework, ~2500 files
  "https://github.com/expressjs/express"      # JavaScript — web framework, ~50 files
  "https://github.com/sindresorhus/got"       # TypeScript — HTTP client, ~50 files
  "https://github.com/gin-gonic/gin"          # Go — web framework, ~100 files
  "https://github.com/actix/actix-web"        # Rust — web framework, ~150 files
  "https://github.com/oakserver/oak"          # TypeScript (Deno) — web framework, ~50 files
  "https://github.com/encode/httpx"           # Python — async HTTP client, ~150 files
)
TIMEOUT=300  # 5 minutes per repo

: > "$LOGFILE"

{
  echo "archex showcase batch — $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "archex version: $(uv run archex --version)"
  echo "budget: $BUDGET"
  echo "repos: ${#REPOS[@]}"
  echo ""
} | tee "$LOGFILE"

failed=0
for repo in "${REPOS[@]}"; do
  name="${repo##*/}"
  echo "[$name] running..." | tee -a "$LOGFILE"
  if perl -e "alarm $TIMEOUT; exec @ARGV" -- uv run python scripts/showcase.py "$repo" --budget "$BUDGET" >> "$LOGFILE" 2>&1; then
    echo "[$name] done" | tee -a "$LOGFILE"
  else
    rc=$?
    if [ "$rc" -eq 142 ]; then
      echo "[$name] TIMEOUT after ${TIMEOUT}s" | tee -a "$LOGFILE"
    else
      echo "[$name] FAILED (exit $rc)" | tee -a "$LOGFILE"
    fi
    ((failed++))
  fi
  echo "" >> "$LOGFILE"
done

{
  echo "═══════════════════════════════════════════════════════════════"
  echo "  Batch complete: $((${#REPOS[@]} - failed))/${#REPOS[@]} succeeded"
  echo "  Log: $LOGFILE"
  echo "═══════════════════════════════════════════════════════════════"
} | tee -a "$LOGFILE"
