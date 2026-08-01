#!/usr/bin/env bash
set -euo pipefail

DEST=${1:?usage: bootstrap_robomemarena.sh DESTINATION}
COMMIT=514ecdf86ba47d496ab1728a827670833107ffd3
REPO=https://github.com/OpenHelix-Team/RoboMemArena.git

export all_proxy=${all_proxy:-socks5://localhost:9632}
export ALL_PROXY=${ALL_PROXY:-$all_proxy}

if [[ -e "${DEST}" && ! -d "${DEST}/.git" ]]; then
  echo "destination exists but is not a Git checkout: ${DEST}" >&2
  exit 2
fi
if [[ ! -d "${DEST}/.git" ]]; then
  git clone "${REPO}" "${DEST}"
fi
git -C "${DEST}" fetch --tags origin
git -C "${DEST}" checkout --detach "${COMMIT}"
test "$(git -C "${DEST}" rev-parse HEAD)" = "${COMMIT}"
test -f "${DEST}/evaluation_benchmark/scripts/task2_26_reference_stage.py"
printf 'RoboMemArena ready at %s\n' "${DEST}"
