"""
NVRD-style higher-level perturbations of the affordance-guided objects, 5 levels each, with gpt-image-2.6-sunburst
(image edits via OpenRouter's /images endpoint, `input_references`) and gpt-6-astra as the quality judge.

Perturbations (nvrd.pdf appendix C.2; background replacement left out):
  shape_deformation  NVRD's prompt, chained: level k edits level k-1
  part_addition      NVRD's prompt, chained: one more prominent part per level
  part_removal       gpt-6-astra plans 5 large chunks to erase so ~20/40/60/75/90% of the object is gone after
                     each level (any part, the working one included); each level removes the next chunk;
                     level k removes part k from level k-1, told which parts are already gone
  style_degradation  5 of NVRD's 20 trajectory steps (STYLE_STEPS), chained
  texture_shift      one edit putting a target texture on the object (NVRD's shape-texture textures),
                     then a pixel blend original -> textured at t = 0.2, 0.4, ... 1.0 (as NVRD interpolated)
Unlike NVRD, every prompt keeps the object usable for its task ("... without removing its ability to be
used for <task>", the phrasing of the user's shape test), part removal only removes parts the task doesn't
need. The judge does not check usability (user's call): only that the edit is visible, the image clean
and the step continuous.

Every generative edit is checked by gpt-6-astra (shown the previous and the new image): the edit must be
clearly visible, the image still one object on a plain white background with no text, and the step not
a wholesale replacement of the object. A failing edit is retried up to MAX_TRIES times; the last try is
kept either way and flagged in the manifest.

Inputs:  data/<id>/<id>.png (the original object)
Outputs: data/<id>/<folder>/<k>.png, k = 1..5, folder = deform | add | remove | style | texture (FOLDERS);
         texture level 5 is the full-strength textured image (blend t = 1.0)
         json/affordance_guided_perturbations.json (prompts, judge verdicts, part lists, texture)
Usage (from nvrd/affordance-test/):
    OPENROUTER_API_KEY=... python3 code/image_generation/perturb_objects.py scoop_stir_soup
    ... --only shape_deformation,part_removal scoop_stir_soup
"""
import base64
import io
import json
import os
import sys
import time

import requests
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # affordance-test
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from generate_affordance_guided import SCENARIOS  # noqa: E402

DATA = os.path.join(HERE, "data")  # data/<id>/<id>.png + data/<id>/<folder>/<1..5>.png
FOLDERS = {"shape_deformation": "deform", "part_addition": "add", "part_removal": "remove",
           "style_degradation": "style", "texture_shift": "texture"}
MANIFEST = os.path.join(HERE, "json", "affordance_guided_perturbations.json")
IMAGE_MODEL = os.environ.get("OPENROUTER_IMAGE_MODEL", "openai/gpt-image-2.6-sunburst")  # newest gpt-image
QUALITY = os.environ.get("PERTURB_QUALITY", "high")
JUDGE_MODEL = os.environ.get("PERTURB_JUDGE_MODEL", "openai/gpt-6-astra")
LEVELS = 5
MAX_TRIES = int(os.environ.get("PERTURB_MAX_TRIES", "3"))
BASE = "https://openrouter.ai/api/v1"

KEEP_WHITE = " Keep the plain white background, show only this one object, and add no text."

# Function-preserving versions of NVRD's prompts (2026-11-23): NVRD's shape prompt melted the ladle into a
# holey mass with no bowl by level 3. Phrasing from the user's test trial: "Visually deform this object
# without removing its ability to scoop and stir liquids".
KEEP_FUNCTION = " without removing its ability to be used for {task}"
# One-step tests from the user's ladle (2026-11-23): the user's plain prompt barely moved gpt-image-2 or 2.6-flare;
# 2.6-sunburst with this stronger wording matched the deformation of the user's own ChatGPT level 1.
# Toned down a step (user, 2026-11-23: "tone down the shape deformation intensity just a bit"); the stronger
# "dramatically ... clearly and substantially different" version had holes in the bowl by level 3.
SHAPE_PROMPT = ("Visually deform this object noticeably: bend, twist, stretch and reshape it, the part that does the "
                "work included, so its shape is visibly different from the current image," + KEEP_FUNCTION + ".")
ADD_PROMPT = "Add a large, clearly visible new part to this object" + KEEP_FUNCTION + "."
# Part removal (2026-11-23, user: "by level 5, 90+% of the object is gone"): the earlier "parts NOT needed for the
# task" list only offered small handle knobs, so 5 levels barely changed the ladle. Now the plan removes large
# chunks, working part included, so the object shrinks to ~REMOVE_TARGETS[k] gone after level k.
REMOVE_TARGETS = [20, 40, 60, 75, 90]
REMOVE_PROMPT = ("Remove the {part} from this object. Completely erase it so there is a clear, visible gap or absence "
                 "where it used to be, and leave the rest of the object exactly as it is. After this step about "
                 "{pct}% of the ORIGINAL object should be gone in total. Parts already removed in previous steps: "
                 "{done}; do not bring them back.")
PART_LIST_PROMPT = (
    "Analyze this image of a tool for {task} carefully. Plan how to erase it from the image in {n} steps, so that "
    "after step 1, 2, 3, 4, 5 about {targets}% of the object's visible area is gone in total; after the last step "
    "only a small fragment (about one tenth of the object) remains.\nRules:\n"
    "- Each step removes one large, concrete chunk: a named part or region (e.g. \"the whole handle above the second "
    "opening\", \"the left half of the bowl\", \"the bowl's rim and front wall\"), sized to reach that step's total.\n"
    "- Any part may go, including the part that does the job.\n"
    "- Be specific about LEFT vs RIGHT, FRONT vs BACK, TOP vs BOTTOM, upper vs lower.\n"
    "- Each step must remove something not removed before; the fragment left at the end must be part of the original.\n"
    'Respond with JSON only: {{"parts": ["...", ...], "remaining": "what is left after the last step"}}')
STYLE_PROMPT = ("Change the art style of this image so it looks like {desc}, keeping the same object, pose and "
                "composition, while it still clearly shows an object that could be used for {task}.")
# NVRD's 20-step trajectory (Table 1); levels use steps 2, 5, 8, 11, 14.
STYLE_STEPS = {
    2: "a professional digital illustration; smoother color transitions, slightly less realistic lighting",
    5: "an illustration with noticeably flat colors and slightly imprecise line work",
    8: "a bold, blocky graphic using only about 6-8 large flat color regions",
    11: "a very rough single-color sketch; a few thick imprecise strokes",
    14: "2-3 thick, rough smears of dark paint in approximately the original shape",
}
TEXTURES = ["bird feathers", "checkerboard", "crochet", "slime"]  # NVRD shape-texture textures + its slime example
TEXTURE_PROMPT = ("Change the surface texture of this object to {texture}" + KEEP_FUNCTION + ", keeping its shape, "
                  "structure, proportions and pose the same. Keep the plain white background.")
JUDGE_PROMPT = (
    "You are checking one step of a gradual image-edit sequence. The FIRST image is the previous step, the SECOND is "
    "the result of this edit: \"{edit}\".\nAnswer each strictly:\n"
    "1. applied: the requested edit is clearly and visibly applied in the second image (not a negligible change).\n"
    "2. clean: the second image shows a single object on a plain white background, with no text, labels or extra "
    "objects.\n3. continuous: the second image is recognisably an edited version of the first (same overall object "
    "and pose), not a completely different object.\n"
    'Respond with JSON only: {{"applied": true, "clean": true, "continuous": true, "reason": "one sentence"}}')
PERTURBATIONS = ["shape_deformation", "part_addition", "part_removal", "style_degradation", "texture_shift"]


def key() -> str:
    return os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def edit_image(png: bytes, prompt: str) -> bytes:
    body = {"model": IMAGE_MODEL, "prompt": prompt, "quality": QUALITY, "aspect_ratio": "1:1", "n": 1,
            "input_references": [{"type": "image_url", "image_url": {"url": data_url(png)}}]}
    last = None
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}/images", json=body, timeout=300, headers={"Authorization": f"Bearer {key()}"})
            if r.status_code == 200:
                return base64.b64decode(r.json()["data"][0]["b64_json"])
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (400, 401, 402, 403):
                break
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(last)


def ask(images: list, text: str) -> dict:
    content = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in images] + [{"type": "text", "text": text}]
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}/chat/completions", timeout=300, headers={"Authorization": f"Bearer {key()}"},
                              json={"model": JUDGE_MODEL, "max_tokens": 4000,
                                    "messages": [{"role": "user", "content": content}]})
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"].get("content") or ""
                return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5 * (attempt + 1))
    return {}


def judged_edit(prev: bytes, prompt: str, edit_desc: str, task: str) -> tuple:
    """Edit with up to MAX_TRIES tries; returns (png, verdict, tries)."""
    out, verdict = None, {}
    for t in range(1, MAX_TRIES + 1):
        out = edit_image(prev, prompt)
        verdict = ask([prev, out], JUDGE_PROMPT.format(edit=edit_desc, task=task))
        if all(verdict.get(k) for k in ("applied", "clean", "continuous")):
            return out, {**verdict, "ok": True}, t
        print(f"      try {t} rejected: {verdict.get('reason')}", flush=True)
    return out, {**verdict, "ok": False}, MAX_TRIES


def blend(a: bytes, b: bytes, t: float) -> bytes:
    ia, ib = Image.open(io.BytesIO(a)).convert("RGB"), Image.open(io.BytesIO(b)).convert("RGB")
    ib = ib.resize(ia.size)
    buf = io.BytesIO()
    Image.blend(ia, ib, t).save(buf, format="PNG")
    return buf.getvalue()


def save_level(perturbation: str, oid: str, level: int, png: bytes) -> str:
    """data/<id>/<folder>/<level>.png, level 1..5."""
    return save(os.path.join(DATA, oid, FOLDERS[perturbation]), str(level), png)


def save(obj_dir: str, name: str, png: bytes) -> str:
    os.makedirs(obj_dir, exist_ok=True)
    path = os.path.join(obj_dir, f"{name}.png")
    with open(path, "wb") as f:
        f.write(png)
    return os.path.relpath(path, HERE)


def run_object(oid: str, only: list, texture_index: int) -> dict:
    task = SCENARIOS[oid][0]
    src = open(os.path.join(DATA, oid, f"{oid}.png"), "rb").read()
    rec = {"id": oid, "task": task, "keep_function": True, "image_model": IMAGE_MODEL,
           "quality": QUALITY, "judge_model": JUDGE_MODEL, "perturbations": {}}

    def chain(name, prompts_and_descs):
        prev, levels = src, []
        for k, (prompt, desc) in enumerate(prompts_and_descs, start=1):
            print(f"    {name} L{k}", flush=True)
            png, verdict, tries = judged_edit(prev, prompt + KEEP_WHITE, desc, task)
            levels.append({"level": k, "prompt": prompt, "image": save_level(name, oid, k, png),
                           "judge": verdict, "tries": tries})
            prev = png
        rec["perturbations"][name] = {"levels": levels}

    if "shape_deformation" in only:
        chain("shape_deformation", [(SHAPE_PROMPT.format(task=task), "visibly deform the object's shape")] * LEVELS)
    if "part_addition" in only:
        chain("part_addition", [(ADD_PROMPT.format(task=task), "add a large new part")] * LEVELS)
    if "part_removal" in only:
        plan = ask([src], PART_LIST_PROMPT.format(task=task, n=LEVELS, targets=", ".join(map(str, REMOVE_TARGETS))))
        parts = plan.get("parts", [])[:LEVELS]
        if len(parts) < LEVELS:
            print(f"    part list came back short: {parts}", flush=True)
        print(f"    removal plan: {parts} -> leaves {plan.get('remaining')}", flush=True)
        steps = [(REMOVE_PROMPT.format(part=p, pct=REMOVE_TARGETS[i], done=", ".join(parts[:i]) or "none"),
                  f"remove the {p}, so that about {REMOVE_TARGETS[i]}% of the original object is gone in total")
                 for i, p in enumerate(parts)]
        chain("part_removal", steps)
        rec["perturbations"]["part_removal"]["parts"] = parts
        rec["perturbations"]["part_removal"]["remaining"] = plan.get("remaining")
        rec["perturbations"]["part_removal"]["targets_pct_gone"] = REMOVE_TARGETS
    if "style_degradation" in only:
        chain("style_degradation", [(STYLE_PROMPT.format(desc=d, task=task), f"make it look like {d}")
                                    for _, d in sorted(STYLE_STEPS.items())])
        rec["perturbations"]["style_degradation"]["nvrd_steps"] = sorted(STYLE_STEPS)
    if "texture_shift" in only:
        texture = TEXTURES[texture_index % len(TEXTURES)]
        print(f"    texture_shift ({texture})", flush=True)
        full, verdict, tries = judged_edit(src, TEXTURE_PROMPT.format(texture=texture, task=task),
                                           f"give the object a {texture} texture", task)
        levels = [{"level": k, "blend": k / LEVELS,
                   "image": save_level("texture_shift", oid, k, blend(src, full, k / LEVELS))}
                  for k in range(1, LEVELS + 1)]
        rec["perturbations"]["texture_shift"] = {"texture": texture, "judge": verdict, "tries": tries, "levels": levels}
    return rec


def main():
    args = sys.argv[1:]
    only = PERTURBATIONS
    if args and args[0] == "--only":
        only, args = args[1].split(","), args[2:]
    all_ids = sorted(d for d in os.listdir(DATA) if os.path.isfile(os.path.join(DATA, d, f"{d}.png")))
    ids = args or all_ids
    manifest = {}
    if os.path.isfile(MANIFEST):
        manifest = {r["id"]: r for r in json.load(open(MANIFEST))}
    for oid in ids:
        print(f"=== {oid} ===", flush=True)
        rec = run_object(oid, only, all_ids.index(oid) if oid in all_ids else 0)
        if oid in manifest:  # keep perturbations not rerun this time
            manifest[oid]["perturbations"].update(rec["perturbations"])
            rec["perturbations"] = manifest[oid]["perturbations"]
        manifest[oid] = rec
        json.dump(list(manifest.values()), open(MANIFEST, "w"), indent=2)
    print(f"done -> {DATA}/<id>/<folder>/, {MANIFEST}")


if __name__ == "__main__":
    main()
