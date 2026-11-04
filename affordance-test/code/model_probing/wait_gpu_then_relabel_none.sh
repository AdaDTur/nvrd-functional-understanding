#!/bin/bash
# Same pattern as wait_gpu_then_relabel.sh: waits for at least MIN_FREE_MIB free on every GPU, then
# runs relabel_none_cases.py (targeted re-score of the pilot's "none" regions against the no-none
# vocabulary). Launched via setsid+nohup+disown so it survives a harness session reconnect.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"   # affordance-test (script moved under code/model_probing/)
VENV_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"  # nvrd (venvs moved out of affordance-test)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/relabel_none_cases.log"
MIN_FREE_MIB=10000  # bumped from 5000 after molmo2-8b OOM'd mid-run with 9245MiB reported free --
                     # another process grew on GPU 0 between the wait check and the actual load

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== wait start (pid=$$), waiting for >${MIN_FREE_MIB}MiB free on every GPU ==="
while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt "$MIN_FREE_MIB" ]; then
        log "GPU memory available (min free across GPUs: ${min_free}MiB) -- launching relabel_none_cases.py"
        break
    fi
    sleep 30
done

"$VENV_ROOT/venv_vlm/bin/python3" "$HERE/relabel_none_cases.py" >> "$LOG" 2>&1
log "=== relabel_none_cases.py exited with code $? ==="
