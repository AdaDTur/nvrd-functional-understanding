#!/bin/bash
# Watchdog for batch_final_labeling.py: restarts it if killed or crashed (e.g. the agent session itself
# ending mid-run, which is what killed a Qwen3-VL test run once already), using --resume so a restart
# picks up only the objects that don't have an output yet instead of redoing completed ones.
#
# Launch this DETACHED from the controlling shell/session, not just as an ordinary background job --
# a plain "&" is still a child of that shell's process group and dies with it. Use:
#   cd affordance-test && setsid nohup bash labeling_watchdog.sh <model_key> [object ...] \
#       </dev/null >/dev/null 2>&1 & disown
# setsid detaches it into its own session (immune to the launching shell's signals), nohup ignores
# SIGHUP, and disown removes it from the launching shell's job table -- all three together are what
# let it survive the agent session ending, not any one alone.
#
# Usage: [CUDA_VISIBLE_DEVICES=..] labeling_watchdog.sh <model_key> [--split novel | object ...]
# Logs to logs/watchdog_<model_key>.log. Stops on the first exit-0 (batch_final_labeling.py returns 0
# both for "finished a full pass" and for "--resume found nothing left to do" -- either way, done).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"   # affordance-test (script moved under code/model_probing/)
VENV_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"  # nvrd (venvs moved out of affordance-test)
MODEL_KEY="${1:?usage: labeling_watchdog.sh <model_key> [object ...]}"
shift
MAX_RETRIES=20
RETRY_DELAY_S=15
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
# AFF_VOCAB=synthia_filtered (see batch_final_labeling.py) logs separately from the AGD20K runs
VOCAB_TAG="${AFF_VOCAB:+_${AFF_VOCAB%%_*}}"
LOG="$LOG_DIR/watchdog_${MODEL_KEY}${VOCAB_TAG}.log"
# Each MODELS entry names the venv that can load it ("venv": ...) -- Gemma-4/Qwen3.5 need
# transformers 5.x, Phi-3.5 needs 4.46. Override with VENV=<name>; legacy entries default to venv_vlm.
VENV="${VENV:-$(grep -A4 "^    \"$MODEL_KEY\": {" "$HERE/batch_final_labeling.py" | grep -o '"venv": "[^"]*"' | head -1 | cut -d'"' -f4)}"
VENV="${VENV:-venv_vlm}"

log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

log "=== watchdog start (model=$MODEL_KEY, venv=$VENV, gpus=${CUDA_VISIBLE_DEVICES:-all}, objects=${*:-<all>}, pid=$$, sid=$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ')) ==="
YIELD_EXIT_CODE=75  # batch_final_labeling.py exited to release the GPU to another user's job
attempt=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
    log "attempt $attempt/$MAX_RETRIES: launching batch_final_labeling.py --model $MODEL_KEY --resume $*"
    "$VENV_ROOT/$VENV/bin/python3" "$HERE/batch_final_labeling.py" --model "$MODEL_KEY" --resume "$@" >> "$LOG" 2>&1
    code=$?
    if [ "$code" -eq 0 ]; then
        log "attempt $attempt: exited 0 (done, or --resume found nothing left) -- stopping"
        exit 0
    fi
    if [ "$code" -eq "$YIELD_EXIT_CODE" ]; then
        # Not a crash: it gave the GPU back under contention. Relaunch (not counted as a retry);
        # the new process waits for headroom before loading the model.
        log "yielded GPU to another job -- relaunching, will wait for headroom before loading"
        sleep "$RETRY_DELAY_S"
        continue
    fi
    log "attempt $attempt: exited $code (killed or crashed) -- retrying in ${RETRY_DELAY_S}s"
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY_S"
done
log "giving up after $MAX_RETRIES attempts, still failing"
exit 1
