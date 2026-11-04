"""
Two modes, both via Gemini 2.5 Flash Image ("nano-banana"):

--mode edit (default, the original behavior): for every NVRD novel object and every affordance in
a fixed vocabulary, ask the model to edit the object's image so its form more clearly affords that
action -- e.g. shape_16 edited to grow a cupped, grasping structure for "grasp" -- the same kind of
edit the user hand-tested with Gemini directly (shape_16 -> "cup holder" affordance). A generative
counterpart to the rest of the pipeline: instead of asking a VLM to label which affordance a region
already has, it asks an image model to visibly ADD a given affordance to an EXISTING object.

--mode generate (new): the opposite direction -- no source image, pure text-to-image. Generates an
ENTIRELY NEW novel object from scratch for a target real-world task (e.g. "watering a plant"),
explicitly prompted to have zero resemblance to the canonical real object for that task while still
visibly affording it -- the same prompting pattern the user hand-tested directly against
GPT/Gemini image models (see GENERATE_SCENARIOS below). This exists to test a hypothesis raised
after the synthia_filtered vocabulary run (label_synthia_vocab.py) still came back mostly "none":
maybe that's not a vocabulary problem at all, but a DATA problem -- the existing shape_N novel
objects were procedurally generated with no functional target, so a VLM finding no affordance may
be an accurate read of shapes that never had one. Affordance-GUIDED generation (this mode) produces
a second, comparable novel-object set that, unlike shape_N, has a real functional target baked into
every image by construction, for a head-to-head comparison against the existing 30 novel objects.

No manual environment setup needed -- just:
    python3 generate_affordance_edits.py [object ...]
Run with ANY python3 (even bare system python with nothing installed). The bootstrap block right
below re-execs the process under ./venv (created once via `python3 -m venv venv` + `pip install
openai google-genai pillow` into it) if it's not already running there, and loads GEMINI_API_KEY /
OPENAI_API_KEY from ./.env automatically -- so `python3 <script>.py` is the whole invocation.

Usage:
    python3 generate_affordance_edits.py [object ...]   # edit mode (default): first 5 novel objects
    OBJECT_LIMIT=10 python3 generate_affordance_edits.py                 # edit mode: all 10 local novel objects
    AFFORDANCE_LIMIT=5 python3 generate_affordance_edits.py shape_16     # edit mode: explicit object, 5 affordances
    python3 generate_affordance_edits.py --mode generate                # generate mode: all 30 GENERATE_SCENARIOS
    python3 generate_affordance_edits.py --mode generate watering drain # generate mode: just these scenario ids
Already-generated (object, affordance) pairs -- or, in generate mode, already-generated scenario
ids -- are skipped automatically, so a rerun resumes rather than regenerating; delete the specific
output PNG to redo one.
"""
import os
import sys

# --- self-bootstrap: re-exec under ./venv, then auto-load ./.env -- before any third-party import,
# so this works invoked with a bare system python3 that has nothing installed yet. ---
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
_VENV_DIR = os.path.join(os.path.dirname(HERE), "venv")  # venvs live in nvrd/, one level above affordance-test
# Check sys.prefix, not sys.executable -- on this system venv/bin/python3 is a symlink to the same
# system python *binary*, so comparing executable paths never detects "not yet in the venv"; prefix
# reflects whether the interpreter was actually started as the venv (via its pyvenv.cfg), which is
# what actually matters (site-packages, i.e. whether openai/google-genai/pillow are importable).
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _venv_py = os.path.join(_VENV_DIR, "bin", "python3")
    os.environ["PYTHONUNBUFFERED"] = "1"  # execv drops the original -u flag (argv doesn't carry it)
    os.execv(_venv_py, [_venv_py] + sys.argv)

_ENV_FILE = os.path.join(HERE, ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())  # real exported env vars still win

import base64
import json
import re
import time
import io

from PIL import Image

REPO_ROOT = os.path.dirname(HERE)
NOVEL_DIR = os.path.join(REPO_ROOT, "regions_test", "data", "novel")
OUT_DIR = os.path.join(HERE, "figures", "affordance_edits")
JSON_DIR = os.path.join(HERE, "json", "affordance_edits")
# NOT created here -- this module is imported just for its shared constants/prompts by other
# scripts (e.g. generate_affordance_edits_openai.py) regardless of which mode, if any, actually
# runs; creating these eagerly at import time left empty figures/json folders lying around for
# modes that were never invoked. main_edit()/main_generate() create their own output dir lazily,
# only when that mode actually runs.

# --mode generate's own output locations, kept separate from the edit-mode ones above so the two
# never collide or get mixed up. GENERATE_OUT_DIR is the SAME path generate_affordance_edits_openai
# .py's --mode generate writes to (figures/affordance_guided_novel_objects/base_images/) -- that's
# intentional: that folder is the one canonical "current affordance-guided object set", not a
# per-provider sandbox, so whichever provider's run is kept becomes what's there. The two providers'
# json manifests (GENERATE_JSON_DIR below vs the _openai one) stay in separate directories, so the
# full generation history/attempts for each provider is never lost even though the image files
# themselves share one folder.
GENERATE_OUT_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "base_images")
GENERATE_JSON_DIR = os.path.join(HERE, "json", "affordance_guided_objects")

MODEL_NAME = os.environ.get("AFFORDANCE_EDIT_MODEL", "gemini-2.5-flash-image")
MAX_RETRIES = 3
RETRY_BACKOFF_S = 10
REQUEST_DELAY_S = 2  # be polite to the API between calls

# GAT's AFF_LIST (run_gat_inference.py), with a one-line gloss each -- the 5 shared with
# affordance_labeling.UMD_GLOSSES reuse its exact wording; screw/stick are new, written to match style.
AFFORDANCE_VOCAB = {
    "grasp": "can be picked up or held with the fingers",
    "cut": "has a thin, flat, wide bladed edge along a straight or curved line, for slicing through "
           "something the way a knife blade does -- a flat plane with a sharp LINE of an edge, not a point",
    "scoop": "has a curved surface for scooping material up",
    "pound": "can be used to strike or pound",
    "support": "can hold up other objects",
    "screw": "can be twisted into or onto something",
    "contain": "has a cavity that holds things inside",
    "stick": "has a slender, narrow, tapering POINT for piercing or spearing into something, like a "
             "needle, nail, or thorn tip -- a sharp point, not a wide flat blade",
}

def _natural_key(name: str):
    # "shape_2" before "shape_10" -- plain sorted() puts "shape_10" before "shape_2" (lexicographic).
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


NOVEL_OBJECTS = sorted(
    (os.path.splitext(f)[0] for f in os.listdir(NOVEL_DIR) if f.endswith(".png")), key=_natural_key,
) if os.path.isdir(NOVEL_DIR) else []

_AFF_LIMIT = os.environ.get("AFFORDANCE_LIMIT")
if _AFF_LIMIT:
    AFFORDANCE_VOCAB = dict(list(AFFORDANCE_VOCAB.items())[:int(_AFF_LIMIT)])

# Default to the first 5 objects rather than all of them -- pass explicit object names on the
# command line, or set OBJECT_LIMIT, to run on more/fewer/different ones.
OBJECT_LIMIT = int(os.environ.get("OBJECT_LIMIT", 5))

PROMPT_TEMPLATE = (
    "This image shows an abstract, non-representational 3D-rendered object. Reshape THIS EXACT object "
    "so it clearly and unmistakably affords \"{affordance}\" ({gloss}) -- but aim for a MODERATE, "
    "well-judged amount of change, not an extreme one: change enough that a person would immediately "
    "agree the object could be used to {affordance} something, while changing as little else as "
    "possible. Think of the result as a later model in the same product line as the original -- "
    "obviously designed by the same team, from the same materials and parts, just adapted for a new "
    "purpose -- not a different object.\n\n"
    "Preserve exactly as they are in the original image, pixel-for-pixel where possible: the object's "
    "overall outline/silhouette outside the one feature you reshape for \"{affordance}\", the number and "
    "arrangement of its distinct limbs/rings/protrusions, its material finish and surface texture, its "
    "exact color palette, the lighting direction and shadows, the camera angle, and the background. "
    "Change only what is necessary to add the \"{affordance}\" feature; leave every other part of the "
    "object untouched rather than regenerating the whole object from scratch.\n\n"
    "Do NOT let it become a generic, instantly-recognizable everyday tool (e.g. scissors, a hammer, a "
    "door handle, a screw) that merely borrows the original's colors -- it should still look like a "
    "distinct alien/sci-fi design, not a real-world object."
)


# --mode generate: 30 real-world tasks, one generated novel object each, for a head-to-head
# comparison against the 30 existing procedurally-generated shape_N novel objects. Each entry
# mirrors the exact prompt pattern the user hand-tested directly against GPT/Gemini image models
# (watering a plant / scooping+stirring soup / supporting a human or item): a task, the category of
# real object it should have zero resemblance to, and anything the image should explicitly exclude
# (the thing being acted on, so the object itself -- not a scene -- is what's shown).
# id -> (task, avoid_category, exclude_clause)
GENERATE_SCENARIOS = {
    "water_plant": ("watering a plant", "watering cans", "the actual plant"),
    "scoop_stir_soup": ("scooping and stirring soup", "ladles", "the soup or pot or anything else"),
    "support_human_item": ("supporting a human or item", "seats or ottomans", "any human or other objects or anything else"),
    "cut_food": ("cutting food", "knives", "any food or cutting board"),
    "carry_items": ("carrying groceries or other items by hand", "bags or baskets", "any groceries or other items"),
    "contain_liquid": ("holding and containing liquid for drinking", "cups, mugs, or bottles", "any liquid or beverage"),
    "measure_length": ("measuring the length of an object", "rulers or tape measures", "any object being measured"),
    "illuminate_room": ("illuminating a dark room", "lamps or flashlights", "any room or surroundings"),
    "fasten_papers": ("fastening sheets of paper together", "staplers or paper clips", "any paper"),
    "clean_surface": ("wiping and cleaning a flat surface", "sponges or cloths", "any surface being cleaned"),
    "open_jar_lid": ("opening a tightly sealed jar lid", "jar openers or bottle openers", "any jar or lid"),
    "hang_coat": ("hanging a coat or jacket on a wall", "coat hooks or hangers", "any coat, jacket, or wall"),
    "write_on_paper": ("writing on paper", "pens or pencils", "any paper or writing"),
    "squeeze_juice": ("squeezing juice out of a citrus fruit", "citrus juicers or reamers", "any fruit or juice"),
    "roll_dough": ("rolling out dough flat", "rolling pins", "any dough or flour"),
    "filter_particles": ("straining solid particles out of a liquid", "strainers or sieves", "any liquid or particles"),
    "scoop_ice": ("scooping ice cubes from a bin", "ice scoops or tongs", "any ice or bin"),
    "dry_hands": ("drying wet hands", "towels or hand dryers", "any water or hands"),
    "hold_hot_beverage": ("holding a hot beverage without burning the hand", "mug handles or cup sleeves", "any beverage or liquid"),
    "hook_keys": ("hanging and organizing a set of keys", "key hooks or key racks", "any keys"),
    "prop_up_book": ("propping a book open to a page while reading", "book stands or bookends", "any book"),
    "shield_flame": ("shielding a small flame from wind", "windscreens or lighter guards", "any flame or fire"),
    "anchor_rope": ("anchoring and securing a rope in place", "cleats or stakes", "any rope"),
    "drain_water": ("draining water away from a sink basin", "drain stoppers or plugs", "any sink or water"),
    "grip_pull_door": ("gripping and pulling a heavy door open", "door handles or pull bars", "any door"),
    "amplify_sound": ("amplifying the sound of a phone speaker", "phone speaker docks or amplifier stands", "any phone"),
    "cool_fan_face": ("fanning air to cool a person's face", "hand fans or electric fans", "any person or face"),
    "protect_from_heat": ("protecting a hand from a hot pan while cooking", "oven mitts or pot holders", "any pan or food"),
    "elevate_laptop": ("elevating a laptop to eye level on a desk", "laptop stands", "any laptop or desk"),
    "pound_stake": ("pounding a tent stake into the ground", "mallets or hammers", "any stake or ground"),
}

GENERATE_PROMPT_TEMPLATE = (
    "Generate an image of an entirely novel object/concept that could be used for {task}. Make "
    "sure it has zero resemblance to existing {avoid}, but still has the evident functionality. "
    "Make the background white and add nothing else to the image. Do not include {exclude}."
)

_GEN_LIMIT = os.environ.get("GENERATE_LIMIT")
if _GEN_LIMIT:
    GENERATE_SCENARIOS = dict(list(GENERATE_SCENARIOS.items())[:int(_GEN_LIMIT)])


def resolve_image_path(object_name: str) -> str:
    path = os.path.join(NOVEL_DIR, f"{object_name}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No novel-split image for '{object_name}' at {path}")
    return path


def generate_image(client, task: str, avoid: str, exclude: str, extra: str = ""):
    """Text-to-image, no source image -- returns (PIL.Image | None, text_response | None, error | None).
    Same retry/response-parsing shape as edit_image(), just without an input image in `contents`.
    `extra`: the judge's fix instruction from a prior failed attempt, appended to the prompt verbatim
    (empty string on a first attempt)."""
    from google.genai import types

    prompt = GENERATE_PROMPT_TEMPLATE.format(task=task, avoid=avoid, exclude=exclude) + extra
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            out_image, out_text = None, None
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) is not None:
                    out_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                elif getattr(part, "text", None):
                    out_text = part.text
            if out_image is None:
                last_err = f"no image returned (text: {out_text!r})"
                continue
            return out_image, out_text, None
        except Exception as e:  # noqa: BLE001 -- surfaced to caller, not swallowed
            last_err = str(e)
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return None, None, last_err


def edit_image(client, image: Image.Image, affordance: str, gloss: str):
    """Returns (PIL.Image | None, text_response | None, error | None)."""
    from google.genai import types

    prompt = PROMPT_TEMPLATE.format(affordance=affordance, gloss=gloss)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
            )
            out_image, out_text = None, None
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) is not None:
                    out_image = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                elif getattr(part, "text", None):
                    out_text = part.text
            if out_image is None:
                last_err = f"no image returned (text: {out_text!r})"
                continue
            return out_image, out_text, None
        except Exception as e:  # noqa: BLE001 -- surfaced to caller, not swallowed
            last_err = str(e)
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return None, None, last_err


def main_edit():
    objects = sys.argv[1:] if len(sys.argv) > 1 else NOVEL_OBJECTS[:OBJECT_LIMIT]
    if not objects:
        print(f"No novel objects found under {NOVEL_DIR}")
        return
    print(f"Objects ({len(objects)}): {objects}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set -- export it (https://aistudio.google.com/apikey) and rerun.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    manifest = []
    for obj in objects:
        try:
            img_path = resolve_image_path(obj)
        except FileNotFoundError as e:
            print(f"  {obj}: {e}")
            continue
        image = Image.open(img_path).convert("RGB")
        obj_out_dir = os.path.join(OUT_DIR, obj)
        os.makedirs(obj_out_dir, exist_ok=True)

        print(f"\n=== {obj} ===")
        for affordance, gloss in AFFORDANCE_VOCAB.items():
            out_path = os.path.join(obj_out_dir, f"{obj}_{affordance}.png")
            record = {
                "object": obj, "affordance": affordance, "gloss": gloss, "model": MODEL_NAME,
                "prompt": PROMPT_TEMPLATE.format(affordance=affordance, gloss=gloss),
                "source_image": img_path, "output_image": out_path,
            }
            if os.path.isfile(out_path):
                print(f"  {affordance}: already done, skipping")
                record["status"] = "skipped_existing"
                manifest.append(record)
                continue

            out_image, out_text, err = edit_image(client, image, affordance, gloss)
            if out_image is None:
                print(f"  {affordance}: FAILED -- {err}")
                record["status"], record["error"] = "failed", err
            else:
                out_image.save(out_path)
                print(f"  {affordance}: saved {out_path}" + (f"  (model also said: {out_text!r})" if out_text else ""))
                record["status"] = "ok"
                if out_text:
                    record["model_text"] = out_text
            manifest.append(record)
            time.sleep(REQUEST_DELAY_S)

    manifest_path = os.path.join(JSON_DIR, "manifest.json")
    existing = []
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
    existing = [r for r in existing if not any(r["object"] == m["object"] and r["affordance"] == m["affordance"] for m in manifest)]
    with open(manifest_path, "w") as f:
        json.dump(existing + manifest, f, indent=2)
    print(f"\nSaved {manifest_path}")


# Judge for --mode generate, run after every generation: checks not just "does this look
# thematically related to the task" but whether the shape would ACTUALLY, PHYSICALLY work for it --
# e.g. a "scoop" whose bowl is an open wire cage would let soup pour straight through; it might
# still be a valid stirrer, but it's not a valid scoop, and a surface-level "looks like a kitchen
# tool for soup" check would pass it anyway. Provider-agnostic: judges a PNG on disk regardless of
# which model (Gemini or gpt-image-2) generated it, via OpenRouter's OpenAI-compatible
# chat/completions endpoint -- the same shape generate_affordance_edits_openai.py's edit-mode judge
# already uses. Deliberately NOT gpt-4o-mini ("lightweight" in the cost sense): that script's own
# edit-mode judge comparison found gpt-4o-mini missing real visual/physical errors outright (2 of 3
# on a direct side-by-side), while gpt-6-astra caught them at ~9x the cost but still cents per
# image -- the same tradeoff applies here, arguably more so, since catching an unusable-shape error
# like the soup-scoop one requires real physical reasoning about the geometry, not just vibes.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = os.environ.get("GENERATE_JUDGE_MODEL", "openai/gpt-6-astra")
MAX_JUDGE_ATTEMPTS = 3  # generate+judge rounds per scenario; each sees the previous round's fix

GENERATE_JUDGE_PROMPT = (
    "You are reviewing an AI-generated image of a proposed novel object for a research dataset on "
    "object affordances. These images will be shown to human raters and to vision-language models "
    "to collect affordance judgments, so what matters is whether the object, if it existed as a "
    "real physical thing, would ACTUALLY, PRACTICALLY work for its intended task -- not just "
    "whether it superficially looks thematically related to it.\n\n"
    "The image shows a proposed novel object, intended for: {task}. Answer two questions about it:\n\n"
    "1. NOVELTY: Does it have genuinely minimal resemblance to an existing real-world {avoid} -- "
    "would an ordinary person NOT immediately call it one of those (a generic, instantly-"
    "recognizable version merely restyled or recolored doesn't count as novel)?\n\n"
    "2. PRACTICAL VALIDITY: If this object existed as a real physical thing made of the material "
    "shown, would it ACTUALLY, MECHANICALLY work to accomplish \"{task}\"? Reason concretely about "
    "physics -- e.g. does a container/scoop have continuous, gap-free walls that would hold its "
    "contents, or does it have holes/slats/tines a liquid or small items would fall straight "
    "through? Does a cutting edge have a genuine continuous sharp line, not just a point or a "
    "decorative texture? Does a support structure have real load-bearing contact with the ground "
    "or wall? Is there anything about the geometry that would make it fail at the SPECIFIC stated "
    "task despite looking thematically appropriate at a glance?\n\n"
    "Respond with EXACTLY these lines:\n"
    "Line 1: PASS if the answer to BOTH questions is yes, otherwise FAIL.\n"
    "Line 2: one short sentence explaining why -- if it's a practical-validity failure, name the "
    "specific physical/mechanical problem (e.g. \"the scooping end is an open wire cage, so liquid "
    "would pour straight through it\").\n"
    "Line 3 (ONLY if FAIL): one short, CONCRETE instruction to append to the generation prompt that "
    "would fix exactly this problem on a retry (name the specific feature to add/change, not a "
    "vague adjective). Omit line 3 if PASS."
)
GENERATE_JUDGE_HISTORY_HEADER = (
    "\n\nEarlier attempts at this SAME object/task, for calibration -- do not just repeat or "
    "intensify a direction that already overcorrected:\n"
)

_JUDGE_LINE_PREFIX_RE = re.compile(r"^(line\s*\d+\s*[:.)-]\s*)", re.IGNORECASE)


def parse_generate_judge_response(text: str) -> tuple:
    """Returns (verdict, reasoning, prompt_fix). Same robust parsing as generate_affordance_edits_openai
    .py's edit-mode judge (duplicated here, not imported -- that script already imports FROM this
    one, so importing the reverse direction would be a circular import)."""
    lines = [_JUDGE_LINE_PREFIX_RE.sub("", l).strip() for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if l]
    verdict_idx, verdict = None, "fail"
    for i, line in enumerate(lines):
        token = line.strip().upper().rstrip(".:")
        if token in ("PASS", "FAIL") or token.startswith("PASS") or token.startswith("FAIL"):
            verdict_idx, verdict = i, ("pass" if token.startswith("PASS") else "fail")
            break
    if verdict_idx is None:  # no explicit verdict line found at all -- treat as fail, keep raw text
        return "fail", text.strip(), None
    rest = lines[verdict_idx + 1:]
    reasoning = rest[0] if rest else lines[verdict_idx]
    prompt_fix = rest[1] if verdict == "fail" and len(rest) > 1 else None
    return verdict, reasoning, prompt_fix


def judge_generated(openrouter_api_key: str, image_path: str, task: str, avoid: str, history=None):
    """Returns (verdict: 'pass'|'fail'|'error', reasoning: str, prompt_fix: str|None)."""
    from openai import OpenAI
    client = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL)

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")
    prompt = GENERATE_JUDGE_PROMPT.format(task=task, avoid=avoid)
    if history:
        prompt += GENERATE_JUDGE_HISTORY_HEADER + "\n".join(
            f"  Attempt {i}: {v.upper()} -- {r}" + (f" (then tried: {pf})" if pf else "")
            for i, (v, r, pf) in enumerate(history, start=1)
        )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
                max_tokens=150,
            )
            return parse_generate_judge_response(response.choices[0].message.content.strip())
        except Exception as e:  # noqa: BLE001 -- surfaced to caller, not swallowed
            last_err = str(e)
            print(f"    judge attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
    return "error", last_err, None


def generate_and_judge(gen_client, openrouter_api_key: str, scenario_id: str, task: str, avoid: str,
                        exclude: str, out_path: str) -> dict:
    """Generate, judge, and -- on FAIL -- retry with the judge's own fix appended to the prompt, up
    to MAX_JUDGE_ATTEMPTS rounds. Mirrors generate_affordance_edits_openai.py's generate_and_judge()
    for edit mode, adapted for a single generated image instead of an original/edit pair."""
    extra, attempts, history = "", [], []
    verdict, reasoning = "fail", None
    for attempt_n in range(1, MAX_JUDGE_ATTEMPTS + 1):
        prompt = GENERATE_PROMPT_TEMPLATE.format(task=task, avoid=avoid, exclude=exclude) + extra
        out_image, out_text, err = generate_image(gen_client, task, avoid, exclude, extra=extra)
        if out_image is None:
            attempts.append({"attempt": attempt_n, "prompt": prompt, "error": err})
            print(f"    attempt {attempt_n}/{MAX_JUDGE_ATTEMPTS}: generation FAILED -- {err}")
            time.sleep(REQUEST_DELAY_S)
            continue

        out_image.save(out_path)
        verdict, reasoning, prompt_fix = judge_generated(openrouter_api_key, out_path, task, avoid, history=history)
        attempts.append({
            "attempt": attempt_n, "prompt": prompt, "judge_verdict": verdict,
            "judge_reasoning": reasoning, "judge_prompt_fix": prompt_fix,
        })
        history.append((verdict, reasoning, prompt_fix))
        print(f"    attempt {attempt_n}/{MAX_JUDGE_ATTEMPTS}: judge [{verdict.upper():5s}]: {reasoning}")
        time.sleep(REQUEST_DELAY_S)
        if verdict == "pass":
            break
        if prompt_fix:
            extra = " " + prompt_fix
            print(f"      -> retrying with guidance (replacing prior round's): {prompt_fix}")

    return {
        "status": "ok" if any("judge_verdict" in a for a in attempts) else "failed",
        "judge_model": JUDGE_MODEL, "judge_verdict": verdict, "judge_reasoning": reasoning,
        "attempts": attempts, "retry_exhausted": verdict != "pass",
    }


def main_generate(args: list):
    do_judge = not os.environ.get("GENERATE_NO_JUDGE")
    retry_failed = "--retry-failed" in args
    rejudge = "--rejudge" in args
    args = [a for a in args if a not in ("--retry-failed", "--rejudge")]
    os.makedirs(GENERATE_OUT_DIR, exist_ok=True)
    os.makedirs(GENERATE_JSON_DIR, exist_ok=True)

    scenario_ids = args if args else list(GENERATE_SCENARIOS)
    unknown = [s for s in scenario_ids if s not in GENERATE_SCENARIOS]
    if unknown:
        print(f"Unknown scenario id(s): {unknown}\nChoices: {sorted(GENERATE_SCENARIOS)}")
        sys.exit(1)
    print(f"Scenarios ({len(scenario_ids)}): {scenario_ids}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set -- export it (https://aistudio.google.com/apikey) and rerun.")
        sys.exit(1)
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if do_judge and not openrouter_api_key:
        print("OPENROUTER_API_KEY is not set (needed for the judge) -- add it to .env, or set "
              "GENERATE_NO_JUDGE=1 to skip judging, and rerun.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    manifest_path = os.path.join(GENERATE_JSON_DIR, "manifest.json")
    existing = []
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
    # Keyed dict, saved after every scenario (not just once at the end) -- a kill mid-run only ever
    # loses the ONE scenario in flight, matching this project's other long-running scripts.
    existing_by_id = {r["scenario_id"]: r for r in existing}

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump(list(existing_by_id.values()), f, indent=2)

    manifest = []
    for scenario_id in scenario_ids:
        task, avoid, exclude = GENERATE_SCENARIOS[scenario_id]
        out_path = os.path.join(GENERATE_OUT_DIR, f"{scenario_id}.png")
        record = existing_by_id.get(scenario_id, {})
        record.update({"scenario_id": scenario_id, "task": task, "avoid": avoid, "exclude": exclude,
                        "model": MODEL_NAME, "output_image": out_path})

        is_pass = record.get("judge_verdict") == "pass"
        is_exhausted_fail = record.get("retry_exhausted") and not is_pass
        if do_judge and not rejudge and (is_pass or (is_exhausted_fail and not retry_failed)):
            why = "PASS" if is_pass else "retries exhausted, still FAIL"
            print(f"  {scenario_id}: already resolved ({why}), skipping")
            manifest.append(record)
            continue
        if not do_judge and os.path.isfile(out_path):
            print(f"  {scenario_id}: already done, skipping")
            record["status"] = "skipped_existing"
            existing_by_id[scenario_id] = record
            save_manifest()
            manifest.append(record)
            continue

        print(f"\n=== {scenario_id} ({task}) ===")
        if do_judge:
            if retry_failed and is_exhausted_fail:
                print("  re-attempting previously-exhausted FAIL (--retry-failed)")
            record.update(generate_and_judge(client, openrouter_api_key, scenario_id, task, avoid, exclude, out_path))
        else:
            prompt = GENERATE_PROMPT_TEMPLATE.format(task=task, avoid=avoid, exclude=exclude)
            out_image, out_text, err = generate_image(client, task, avoid, exclude)
            record["prompt"] = prompt
            if out_image is None:
                print(f"  FAILED -- {err}")
                record["status"], record["error"] = "failed", err
            else:
                out_image.save(out_path)
                print(f"  saved {out_path}" + (f"  (model also said: {out_text!r})" if out_text else ""))
                record["status"] = "ok"
                if out_text:
                    record["model_text"] = out_text
            time.sleep(REQUEST_DELAY_S)

        existing_by_id[scenario_id] = record
        save_manifest()
        manifest.append(record)

    print(f"\nSaved {manifest_path}")
    judged = [r for r in manifest if "judge_verdict" in r]
    if judged:
        n_pass = sum(r["judge_verdict"] == "pass" for r in judged)
        print(f"Judge: {n_pass}/{len(judged)} pass")
        for r in judged:
            if r["judge_verdict"] != "pass":
                n_att = len(r.get("attempts", [])) or 1
                print(f"  [{r['judge_verdict'].upper()}] {r['scenario_id']} "
                      f"(after {n_att} attempt{'s' if n_att != 1 else ''}): {r['judge_reasoning']}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--mode":
        mode, args = args[1], args[2:]
    else:
        mode = "edit"
    if mode == "generate":
        sys.argv = sys.argv[:1] + args  # so any code still reading sys.argv directly sees just the remainder
        main_generate(args)
    elif mode == "edit":
        sys.argv = sys.argv[:1] + args
        main_edit()
    else:
        print(f"Unknown --mode {mode!r}; choices: edit, generate")
        sys.exit(1)


if __name__ == "__main__":
    main()
