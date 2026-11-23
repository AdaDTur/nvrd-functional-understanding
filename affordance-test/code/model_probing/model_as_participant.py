"""
Run a vision model through the human study as if it were a participant: for every part of the 30 novel
objects it sees the same two images people see ("All Parts" numbered overlay and "Highlighted Part"),
the same option list in the same (alphabetical) order, and the same instruction (pick up to 3, or write
your own); then the whole-object question with the overall option list. The size chart is left out
(people see one of three sizes depending on their group), so its answers pool over sizes like the
analysis does.

Default model: google/gemini-3.8-flash, a family not used anywhere else in this project (not a study
model, not in the option bank, not a pool-test labeler). Override with PARTICIPANT_MODEL.

Output: json/model_participant_<model>.json  {"model", "parts": [{object, region_id, part, selected, other}],
        "objects": [{object, selected, other}]}  -- read by code/plotting/human_model_alignment.py
Usage (from nvrd/affordance-test/): OPENROUTER_API_KEY=... python3 code/model_probing/model_as_participant.py
"""
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # affordance-test
STUDY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "prolific-novel-concepts", "affordance-study")
MODEL = os.environ.get("PARTICIPANT_MODEL", "google/gemini-3.8-flash")
WORKERS = int(os.environ.get("PARTICIPANT_WORKERS", "8"))
# PARTICIPANT_MAX_PICKS=1: ask for the single best option instead of "up to 3" (people mostly pick one;
# output file gets a _pick1 suffix so both runs can be compared).
MAX_PICKS = int(os.environ.get("PARTICIPANT_MAX_PICKS", "3"))
PICK_TEXT = ("Pick the ONE functionality below that this {what} is most realistically used for"
             if MAX_PICKS == 1 else "Pick up to 3 functionalities below that this {what} could realistically be used for")
URL = "https://openrouter.ai/api/v1/chat/completions"

PART_PROMPT = (
    "You are taking part in a study about new, unfamiliar objects. The first image shows a whole object "
    "divided into {n} numbered parts. The second image shows the same object with part {part} highlighted "
    "and the rest dimmed.\n\n"
    "{pick}, meaning what it "
    "could actually be used to DO. If something is missing, you may write your own instead.\n\n"
    "Options: {options}\n\n"
    'Respond with JSON only: {{"selected": ["option", ...], "other": "your own answer or empty"}}'
)
OBJECT_PROMPT = (
    "You are taking part in a study about new, unfamiliar objects. The image shows a whole object divided "
    "into {n} numbered parts. You have already looked at each part individually; now step back and "
    "consider the object as a whole.\n\n"
    "{pick}, "
    "meaning what it could actually be used to DO. If something is missing, you may write your own instead.\n\n"
    "Options: {options}\n\n"
    'Respond with JSON only: {{"selected": ["option", ...], "other": "your own answer or empty"}}'
)


def img(path: str) -> dict:
    with open(path, "rb") as f:
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(f.read()).decode()}}


def ask(key: str, content: list) -> dict:
    last = None
    for attempt in range(5):
        try:
            r = requests.post(URL, timeout=180, headers={"Authorization": f"Bearer {key}"},
                              json={"model": MODEL, "messages": [{"role": "user", "content": content}],
                                    "response_format": {"type": "json_object"}, "max_tokens": 2000})
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"].get("content") or ""
                text = text.strip().strip("`")
                text = text[text.find("{"):text.rfind("}") + 1]
                return json.loads(text)
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 401, 402, 403):
                break
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(last)


def clean(ans: dict, options: list) -> tuple:
    sel = [s for s in ans.get("selected", []) if isinstance(s, str) and s in options][:MAX_PICKS]
    other = (ans.get("other") or "").strip()
    return sel, other


def main():
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    jobs = []
    for fn in sorted(os.listdir(os.path.join(STUDY, "manifests"))):
        if not fn.startswith("shape_"):
            continue
        obj = fn[:-5]
        man = json.load(open(os.path.join(STUDY, "manifests", fn)))
        n = len(man["regions"])
        numbered = os.path.join(STUDY, "images", f"{obj}_numbered.png")
        for r in man["regions"]:
            opts = r["bank"]
            content = [img(numbered), img(os.path.join(STUDY, "images", f"{obj}_highlight_part{r['part']}.png")),
                       {"type": "text", "text": PART_PROMPT.format(n=n, part=r["part"], options=", ".join(opts),
                                                                 pick=PICK_TEXT.format(what="part"))}]
            jobs.append(("part", obj, r["region_id"], r["part"], opts, content))
        content = [img(numbered), {"type": "text", "text": OBJECT_PROMPT.format(n=n, options=", ".join(man["overall_bank"]),
                                                                pick=PICK_TEXT.format(what="object as a whole"))}]
        jobs.append(("object", obj, None, None, man["overall_bank"], content))
    print(f"{MODEL}: {len(jobs)} questions", flush=True)

    def run(job):
        kind, obj, rid, part, opts, content = job
        try:
            sel, other = clean(ask(key, content), opts)
            if not sel and other:
                sel = []
            if len(sel) >= MAX_PICKS:
                other = ""
            return kind, {"object": obj, "region_id": rid, "part": part, "selected": sel + ([other] if other else []),
                          "other": other}
        except Exception as e:  # noqa: BLE001
            print(f"  {obj} {rid or 'object'}: FAILED {e}", flush=True)
            return kind, None

    with ThreadPoolExecutor(WORKERS) as pool:
        out = list(pool.map(run, jobs))
    parts = [a for k, a in out if k == "part" and a]
    objects = [{k: v for k, v in a.items() if k not in ("region_id", "part")} for k, a in out if k == "object" and a]
    suffix = "" if MAX_PICKS == 3 else f"_pick{MAX_PICKS}"
    path = os.path.join(HERE, "json", f"model_participant_{MODEL.split('/')[-1]}{suffix}.json")
    json.dump({"model": MODEL + suffix.replace("_", " "), "max_picks": MAX_PICKS, "parts": parts,
               "objects": objects}, open(path, "w"), indent=2)
    print(f"saved {len(parts)} parts, {len(objects)} objects -> {path}")


if __name__ == "__main__":
    main()
