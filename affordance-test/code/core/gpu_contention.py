"""
Shared GPU-contention guard for the labeling pipeline, on this shared multi-user machine.

The existing wait_gpu_then_relabel*.sh scripts only check free memory ONCE, before the whole script
starts -- which is exactly what let molmo2-8b OOM on 2026-10-21/22: free memory looked fine when
relabel_none_cases.py launched, but another process grew on GPU 0 while qwen3-vl-8b and
internvl3-8b were still running ahead of it, and molmo2-8b walked straight into the resulting
squeeze with no further check in between.

wait_for_gpu_headroom() is meant to be called repeatedly DURING a run (once per object, or once per
model load), not just once at the top, so a contending job that shows up mid-run gets the same
backoff a pre-launch check would have given it. It watches OTHER processes' own memory footprint,
not just our free/total numbers, so it can react to "someone else's job got big" directly rather than
inferring it after the fact from a free-memory drop -- and so a small, unrelated process doesn't
trigger pausing for no reason.
"""
import os
import subprocess
import sys
import time

MIN_FREE_MIB = 10000  # don't proceed unless every GPU has at least this much free
OTHER_PROC_MIB_THRESHOLD = 8000  # pause if any single GPU's non-us usage exceeds this, even if
                                  # MIN_FREE_MIB is still technically satisfied right now -- this is
                                  # the "especially if it's a larger amount" case, catching a big
                                  # outside job early rather than waiting for it to actually starve us
POLL_SECONDS = 30


def _run_csv(query: str) -> list:
    out = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return [line.split(", ") for line in out.splitlines() if line.strip()]


def gpu_status() -> dict:
    """{gpu_index: {"free_mib": int, "other_mib": int}} -- "other_mib" is every compute process's
    memory on that GPU EXCEPT our own pid (a multi-GPU job shows up as one row per GPU it touches,
    same pid each time, matched by comparing against os.getpid())."""
    free_by_index = {int(idx): int(free) for idx, free in _run_csv("gpu=index,memory.free")}
    uuid_to_index = {uuid: int(idx) for idx, uuid in _run_csv("gpu=index,uuid")}

    other_by_index = {idx: 0 for idx in free_by_index}
    my_pid = os.getpid()
    try:
        rows = _run_csv("compute-apps=pid,used_memory,gpu_uuid")
    except subprocess.CalledProcessError:
        rows = []  # some drivers refuse this query under certain permissions; fail open
    mine_by_index = {idx: 0 for idx in free_by_index}
    for pid, used_mib, uuid in rows:
        if int(pid) == my_pid:
            # Our own footprint counts as available to us: otherwise, once a model is loaded on a
            # 16 GB card, the per-object re-check sees < MIN_FREE_MIB free and blocks on itself.
            idx = uuid_to_index.get(uuid)
            if idx is not None:
                mine_by_index[idx] += int(used_mib)
            continue
        idx = uuid_to_index.get(uuid)
        if idx is not None:
            other_by_index[idx] += int(used_mib)

    # Only the GPUs this process can actually use: with CUDA_VISIBLE_DEVICES=2, another user's job on
    # GPU 0 is irrelevant and would otherwise block forever. (nvidia-smi indices are physical, which is
    # what CUDA_VISIBLE_DEVICES lists; non-integer entries like GPU UUIDs fall back to checking all.)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    indices = list(free_by_index)
    if visible:
        try:
            indices = [int(v) for v in visible.split(",") if v.strip() and int(v) in free_by_index]
        except ValueError:
            pass
    return {idx: {"free_mib": free_by_index[idx] + mine_by_index[idx], "other_mib": other_by_index[idx]}
            for idx in indices}


def contention_blockers(min_free_mib: int = MIN_FREE_MIB,
                        other_proc_mib_threshold: int = OTHER_PROC_MIB_THRESHOLD) -> list:
    """Non-blocking check: one string per visible GPU that's currently contended, [] if clear.
    Same rule wait_for_gpu_headroom() waits on, so a caller can react (e.g. free its own GPU
    memory) before waiting, instead of waiting while still holding the GPU."""
    return [
        f"GPU{idx} free={s['free_mib']}MiB other={s['other_mib']}MiB"
        for idx, s in gpu_status().items()
        if s["free_mib"] < min_free_mib or s["other_mib"] > other_proc_mib_threshold
    ]


def wait_for_gpu_headroom(
    min_free_mib: int = MIN_FREE_MIB,
    other_proc_mib_threshold: int = OTHER_PROC_MIB_THRESHOLD,
    poll_seconds: int = POLL_SECONDS,
    log=lambda msg: print(msg, file=sys.stderr, flush=True),
):
    """Blocks until every GPU has >= min_free_mib free AND no GPU has another process using more
    than other_proc_mib_threshold. Call this before each unit of work (each object, each model
    load), not just once -- that's what makes it catch contention that starts mid-run."""
    waited = False
    while True:
        blockers = contention_blockers(min_free_mib, other_proc_mib_threshold)
        if not blockers:
            if waited:
                log(f"GPU headroom OK, resuming ({gpu_status()})")
            return
        if not waited:
            log(f"Pausing for GPU contention: {blockers}")
        waited = True
        time.sleep(poll_seconds)


if __name__ == "__main__":
    # Quick manual check: `python3 gpu_contention.py` prints current status and exits.
    import json
    print(json.dumps(gpu_status(), indent=2))
