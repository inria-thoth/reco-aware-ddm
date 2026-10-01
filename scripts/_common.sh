set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
PATHS="${PATHS:-default}"
export PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
if [ -n "${DDM4IP_PYTHON_BIN:-}" ]; then export PATH="$DDM4IP_PYTHON_BIN:$PATH"; fi
cd "$REPO_ROOT"
mkdir -p "$RESULTS_DIR"

need() {
    for f in "$@"; do
        [ -e "$f" ] || { echo "missing: $f (run the previous stage first)" >&2; exit 1; }
    done
}

run() {
    local name="$1"; shift
    python ddm4ip/main.py paths="$PATHS" paths.out_path="$RESULTS_DIR" exp_name="$name" "$@" 2>&1 | tee -a "$RESULTS_DIR/$name.console.log"
}
