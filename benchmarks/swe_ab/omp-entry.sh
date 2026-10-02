#!/bin/sh
# omp's entry point inside every task container (RUNBOOK §2), copied into the bundle as
# /opt/omp/bin/omp-entry. Bun's x86-64 build links glibc. On an image without glibc's loader
# (Alpine/musl images: protonmail/webclients and some gravitational/teleport), Bun is started through the glibc loader and
# libraries shipped in the bundle under /opt/omp/glibc; elsewhere it starts directly, exactly as
# before. Only Bun's own process is affected: the tools it runs (sh, git, the repository's test
# commands) are the image's own.
CLI=/opt/omp/install/global/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js
if [ -e /lib64/ld-linux-x86-64.so.2 ]; then
  exec /opt/omp/bin/bun "$CLI" "$@"
fi
exec /opt/omp/glibc/ld-linux-x86-64.so.2 --library-path /opt/omp/glibc /opt/omp/bin/bun "$CLI" "$@"
