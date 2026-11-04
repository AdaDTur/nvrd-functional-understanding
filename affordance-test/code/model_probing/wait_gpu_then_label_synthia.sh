#!/bin/bash
# Same pattern as wait_gpu_then_score_no_none.sh, for label_synthia_vocab.py (all 30 novel objects,
# qwen3-vl-8b, synthia_filtered vocabulary). --resume so a restart after a kill/pause picks up from
# the last object label_synthia_vocab.py hadn't started yet -- it already saves per-part, not just
# per-object, so at most one in-flight part is ever lost.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"   # affordance-test (script moved under code/model_probing/)
VENV_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"  # nvrd (venvs moved out of affordance-test)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/label_synthia_vocab.log"
MIN_FREE_MIB=10000

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== wait start (pid=$$), waiting for >${MIN_FREE_MIB}MiB free on every GPU ==="
while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt "$MIN_FREE_MIB" ]; then
        log "GPU memory available (min free across GPUs: ${min_free}MiB) -- launching label_synthia_vocab.py"
        break
    fi
    sleep 30
done

"$VENV_ROOT/venv_vlm/bin/python3" "$HERE/label_synthia_vocab.py" --resume >> "$LOG" 2>&1
log "=== label_synthia_vocab.py exited with code $? ==="
