#!/bin/bash
# Same pattern as wait_gpu_then_relabel_none.sh, for score_no_none_labels.py (non-destructive
# agd20k_no_none scoring across all 30 novel objects). Launched via setsid+nohup+disown so it
# survives a harness session reconnect.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/score_no_none_labels.log"
MIN_FREE_MIB=10000

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== wait start (pid=$$), waiting for >${MIN_FREE_MIB}MiB free on every GPU ==="
while true; do
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$min_free" -gt "$MIN_FREE_MIB" ]; then
        log "GPU memory available (min free across GPUs: ${min_free}MiB) -- launching score_no_none_labels.py"
        break
    fi
    sleep 30
done

"$HERE/venv_vlm/bin/python3" "$HERE/score_no_none_labels.py" >> "$LOG" 2>&1
log "=== score_no_none_labels.py exited with code $? ==="
