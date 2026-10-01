#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STAGE="${1:-}"; START="${2:-1}"; END="${3:-100}"
[ -n "$STAGE" ] || { echo "usage: $0 step1|step2_diy|step2_ours|step3|oracle [START] [END]" >&2; exit 1; }
for n in $(seq "$START" "$END"); do
    bash "$REPO_ROOT/scripts/super_resolution.sh" "$STAGE" "$n"
done
