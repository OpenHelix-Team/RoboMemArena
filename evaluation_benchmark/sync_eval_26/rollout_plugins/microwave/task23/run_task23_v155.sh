#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# v155 intentionally preserves the v154 evaluator behavior. Fixed-seed repetition
# is implemented by the worker as twenty separate one-episode invocations.
exec bash "${ROOT}/run_task23_v154.sh"
