"""
Affordance-guided novel objects: text-to-image generation with a judge that EDITS THE WHOLE PROMPT
before each retry. The only image-generation script for this project (it replaced
generate_affordance_edits.py, generate_affordance_edits_openai.py and
generate_affordance_only_test.py on 2026-11-10; those live in git history).

Two ways to define an object:
  scenario (default)   a real-world task from SCENARIOS, e.g. "watering a plant", with the real tool
                       category it must not resemble ("watering cans") and what to leave out
  --affordance-only    a bare verb from synthia_affordances_clustered.txt, e.g. "pump"

Each attempt:
  1. generate: openai/gpt-image-2 via OpenRouter's Image API (one square image, white background)
  2. blind look: the judge sees ONLY the image and guesses its use (top 3), before learning the task.
     This is the direct test of whether the function is evident.
  3. review: told the task, the judge scores five criteria, all required to PASS:
       evident  -- the blind guesses name this task (or a clear equivalent)
       correct  -- it would mechanically accomplish this specific task
       specific -- it is built for this task, not a generic vessel/handle/platform that suits many
       correct  is judged as a plausible product concept, not proof of precision (an earlier,
                stricter wording failed every measure_length attempt and piled on mechanism)
       novel    -- the object as a whole is not recognisable as any existing tool (not only the
                scenario's named category); its working part (blade, bristles, pad, spout) may be
                conventional. Forbidding familiar working parts too made every rewrite lose the
                cue that shows the function (0/3 passed in that test round).
       no_text  -- no text, numbers, labels, arrows, icons or screens: shape alone conveys use
                (generated readouts like "248.6 mm" let the blind look pass by reading)
     Rewrites must fix problems with the simplest form and fewest parts.
     On FAIL it returns a complete revised prompt (the previous prompt edited to fix the failing
     criteria, aware of all earlier attempts), which the next attempt uses verbatim.
  4. pool test: three vision models outside the study roster (LABELERS) each pick the object's task
     from 15 options: the intended task plus 14 random others from the pool (the scenario tasks, or
     the synthia vocabulary in --affordance-only mode), shuffled, drawn fresh per labeler.
     Fewer than MIN_POOL_CORRECT (default 1) right = FAIL, and the judge's rewrite must steer the
     object away from the tasks they picked instead.
Only the final image is kept (final.png); every attempt's prompt, blind guesses, pool picks and
scores are recorded in the manifest.

Usage (from affordance-test/; any python3, it re-execs under ../venv and reads OPENROUTER_API_KEY
from .env unless exported):
    python3 code/image_generation/generate_affordance_guided.py                 # scenarios not in KEEP
    python3 code/image_generation/generate_affordance_guided.py cut_food dig_soil
    python3 code/image_generation/generate_affordance_guided.py --affordance-only pump tie
    python3 code/image_generation/generate_affordance_guided.py --primary cut_food dig_soil   # see PRIMARY_PARTS
    V2_MAX_ATTEMPTS=8 ...                                                        # default 6 attempts
Output:
    scenario:        figures/affordance_guided_novel_objects_v2/<id>/final.png (last attempt; earlier ones not kept)
                     json/affordance_guided_objects_v2/manifest.json
    affordance-only: figures/affordance_only_novel_objects/<verb>/..., json/affordance_only_objects/manifest.json
A scenario/verb whose final verdict is already PASS is skipped; delete its manifest entry to redo it.
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
# what actually matters (site-packages, i.e. whether openai/requests are importable).
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

import base64  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import requests  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGES_URL = f"{OPENROUTER_BASE_URL}/images"
IMAGE_MODEL = os.environ.get("OPENROUTER_IMAGE_MODEL", "openai/gpt-image-2")
ASPECT_RATIO = "1:1"  # matches the NVRD renders' native square resolution
QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "high")
# gpt-6-astra over gpt-4o-mini: in a side-by-side the cheaper judge missed real physical errors
# (e.g. calling a visible blade "lacks a sharp edge"); astra caught them at cents per image.
JUDGE_MODEL = os.environ.get("GENERATE_JUDGE_MODEL", "openai/gpt-6-astra")
MAX_ATTEMPTS = int(os.environ.get("V2_MAX_ATTEMPTS", 6))
# Pool test: other models pick the object's task from the whole pool of tasks. Deliberately NOT the
# study's evaluated models (GPT-5.4, Molmo2, Gemma-4, Qwen3.5, Phi-3.5): filtering objects by what
# the study models get right would inflate their benchmark scores. Three different providers.
LABELERS = os.environ.get("POOL_LABELERS",
                          "anthropic/claude-haiku-5.5,x-ai/grok-4.7,mistralai/mistral-medium-3-5").split(",")
MIN_POOL_CORRECT = int(os.environ.get("MIN_POOL_CORRECT", 1))  # attempt fails if fewer labelers pick the right task
HTTP_RETRIES, HTTP_BACKOFF_S = 3, 10

# id -> (task, real tool category it must not resemble, what to leave out of the image)
SCENARIOS = {
    "water_plant": ("watering a plant", "watering cans", "the actual plant"),
    "scoop_stir_soup": ("scooping and stirring soup", "ladles", "the soup or pot or anything else"),
    "support_human_item": ("supporting a human or item", "seats or ottomans", "any human or other objects or anything else"),
    "cut_food": ("cutting food", "knives", "any food or cutting board"),
    "carry_items": ("carrying groceries or other items by hand", "bags or baskets", "any groceries or other items"),
    "dig_soil": ("digging a small hole in garden soil", "trowels or small shovels", "any soil, dirt, or plants"),
    "measure_length": ("measuring the length of an object", "rulers or tape measures", "any object being measured"),
    "comb_hair": ("detangling and combing hair", "combs or hairbrushes", "any hair or person"),
    "scratch_back": ("scratching an itch in the middle of one's own back", "back scratchers", "any person or back"),
    "clean_surface": ("wiping and cleaning a flat surface", "sponges or cloths", "any surface being cleaned"),
    "open_jar_lid": ("opening a tightly sealed jar lid", "jar openers or bottle openers", "any jar or lid"),
    "pick_up_litter": ("picking up litter from the ground without bending down", "grabber tools or tongs", "any litter or ground"),
    "write_on_paper": ("writing on paper", "pens or pencils", "any paper or writing"),
    "squeeze_juice": ("squeezing juice out of a citrus fruit", "citrus juicers or reamers", "any fruit or juice"),
    "roll_dough": ("rolling out dough flat", "rolling pins", "any dough or flour"),
    "filter_particles": ("straining solid particles out of a liquid", "strainers or sieves", "any liquid or particles"),
    "scrape_ice": ("scraping ice off a car windshield", "ice scrapers", "any car, windshield, or ice"),
    "crush_garlic": ("crushing garlic cloves", "garlic presses", "any garlic or food"),
    "funnel_liquid": ("pouring liquid into a narrow-necked bottle without spilling", "funnels", "any liquid or bottle"),
    "hook_keys": ("hanging and organizing a set of keys", "key hooks or key racks", "any keys"),
    "prop_up_book": ("propping a book open to a page while reading", "book stands or bookends", "any book"),
    "sand_wood": ("sanding a rough wooden surface smooth", "sandpaper or sanding blocks", "any wood or surface"),
    "insert_screw": ("driving a screw into a piece of wood", "screwdrivers or drills", "any screw or wood"),
    "peel_vegetable": ("peeling the skin off a potato", "vegetable peelers", "any potato, vegetable, or peel"),
    "grip_pull_door": ("gripping and pulling a heavy door open", "door handles or pull bars", "any door"),
    "squeeze_toothpaste": ("squeezing the last toothpaste out of a tube", "tube squeezers or wringers", "any toothpaste or tube"),
    "sharpen_pencil": ("sharpening a pencil", "pencil sharpeners", "any pencil"),
    "crack_nuts": ("cracking open walnut shells", "nutcrackers", "any nuts or shells"),
    "dispense_tape": ("tearing off a strip of adhesive tape", "tape dispensers", "any tape"),
    "hammer_nail": ("hammering a nail into a wooden board", "hammers or mallets", "any nail or wood"),
    "pump_air": ("pumping air into a bicycle tire", "bicycle pumps or air pumps", "any tire or bicycle"),
}
# Objects kept after the user's curation; they are not regenerated by default.
# Their images are in figures/affordance_guided_novel_objects/base_images/.
KEEP = ["comb_hair", "prop_up_book", "crack_nuts", "funnel_liquid", "roll_dough", "scratch_back",
        "support_human_item", "water_plant", "pump_air"]
# pump_air came from the --affordance-only test (verb "pump"; judged PASS on attempt 2, record in
# json/affordance_only_test/manifest.json) and was added to the kept set on 2026-11-11; its image is
# base_images/pump_air.png (copied from the affordance-only run's pump.png;
# that run's figures folder no longer exists).

SCENARIO_PROMPT = (
    "Generate an image of an entirely novel object/concept that could be used for {task}. Make "
    "sure it has zero resemblance to existing {avoid}, but still has the evident functionality. "
    "Make the background white and add nothing else to the image. Do not include {exclude}. "
    "Show no text, numbers, labels, arrows, icons, logos or screens anywhere on the object; its function must come from its shape alone."
)
# --primary mode (2026-11-13): build the object AROUND a conventional working part, novel everywhere
# else. The user's hand-made comb_hair (familiar tines on an unconventional handle and colour) showed
# the function stays evident when the part that DOES the task is familiar and only the rest is new;
# the judge loop had been failing because it pushed that working part away too. id -> primary part.
PRIMARY_PARTS = {
    "scoop_stir_soup": "a deep, rounded bowl for scooping liquid",
    "cut_food": "a sharp, straight blade edge",
    "carry_items": "a sturdy hand grip that can bear a heavy load, with hooks to hang bags from",
    "dig_soil": "a pointed, scoop-shaped digging blade",
    "measure_length": "a straight edge with evenly spaced tick marks (marks only, no numbers)",
    "clean_surface": "a broad, flat, soft wiping pad",
    "open_jar_lid": "a round gripping ring lined with grippy rubber that fits around a lid",
    "pick_up_litter": "a pair of jaws at the far end of a long shaft that pinch shut",
    "write_on_paper": "a fine writing tip like a pen nib",
    "squeeze_juice": "a ridged, cone-shaped reamer",
    "filter_particles": "a fine mesh screen",
    "scrape_ice": "a flat, straight, rigid scraping edge",
    "crush_garlic": "a perforated chamber with a plunger that presses into it",
    "hook_keys": "a row of small upturned hooks",
    "sand_wood": "a flat surface covered in coarse, gritty abrasive",
    "insert_screw": "a cross-shaped screwdriver tip",
    "peel_vegetable": "a short, slotted peeling blade",
    "grip_pull_door": "a thick bar that the fingers can wrap around, held off the surface by two posts",
    "squeeze_toothpaste": "two parallel rollers with a narrow gap between them",
    "sharpen_pencil": "a cone-shaped hole with a small straight blade along its side",
    "dispense_tape": "a small serrated cutting edge",
    "hammer_nail": "a flat, heavy metal striking face",
}
PRIMARY_PROMPT = (
    "Generate an image of an entirely novel tool that could be used for {task}. Its working part, "
    "{primary}, is what does the task: it must look and work exactly like that familiar part. "
    "Everything else, meaning the handle, body, how the parts connect and the overall silhouette, "
    "must be an unconventional design with zero resemblance to existing {avoid}. Give the tool an "
    "unexpected colour and surface texture or material, chosen so that the working part still does "
    "the task just as well (do not remove, shrink, blunt or weaken it). "
    "Make the background white and add nothing else to the image. Do not include {exclude}. "
    "Show no text, numbers, labels, arrows, icons, logos or screens anywhere on the object."
)
PRIMARY_REVIEW_NOTE = (
    "\n\nThis object is deliberately built around a conventional working part ({primary}). "
    "In revised_prompt, keep that part conventional and clearly recognisable; change only the rest "
    "(handle, body, connections, silhouette, colour, surface texture or material) to fix the failing "
    "criteria, and never in a way that removes, shrinks, blunts or weakens the working part."
)

AFFORDANCE_ONLY_PROMPT = (
    "Generate an image of an entirely novel object that can be used to {task}. "
    "Make the background white and add nothing else to the image. Show only the object itself. "
    "Show no text, numbers, labels, arrows, icons, logos or screens anywhere on the object; its function must come from its shape alone."
)

BLIND_PROMPT = (
    "This image shows one object on a white background. You have not been told what it is for. "
    "Judging only from its shape, parts and materials, what is this object most likely used for? "
    "Give your three best guesses for its primary function, most likely first, each as a short "
    'everyday task phrase (e.g. "slicing bread"). Respond with JSON only: '
    '{"guesses": ["...", "...", "..."], "why": "one sentence on the features you relied on"}'
)
REVIEW_PROMPT = (
    "You are curating AI-generated images of NOVEL objects for a research dataset. Each object must "
    "afford ONE intended task so clearly that a person or a vision-language model, shown only the "
    "image, would infer that task. A separate blind look at this exact image (made without knowing "
    "the task) guessed its uses as: {guesses}.\n\n"
    "Intended task: {task}. It must not resemble existing {avoid}.\n"
    "Prompt that produced this image:\n<<<{prompt}>>>\n\n"
    "Judge five criteria, each true or false with a one-sentence reason:\n"
    "1. evident: the blind guesses name the intended task or a clear equivalent (ideally first).\n"
    "2. correct: there is a plausible physical way it could do the intended task, with no obvious "
    "flaw that would stop it working (gaps that would leak, no edge that could cut, nothing that "
    "could grip, no load-bearing contact, and so on). Do not demand proof of precision or "
    "reliability; judge it the way a person would judge a sensible product concept.\n"
    "3. specific: its form is specialised for this task, not a generic vessel, handle, platform or "
    "blob that would afford many different tasks about equally well.\n"
    "4. novel: judged on the object AS A WHOLE, an ordinary person would NOT immediately "
    "recognise it as any existing tool or product, including but not limited to {avoid} (a "
    "restyled or recoloured version of an existing product does not count as novel). Its WORKING "
    "PART may be conventional (a blade edge, bristles, a gripping pad, a spout, graduated marks) "
    "when that is what makes the function evident; what must be new is the overall object built "
    "around it.\n"
    "5. no_text: the object shows no text, numbers, labels, arrows, icons, logos or screens; the "
    "function is conveyed by shape alone.\n\n"
    "verdict is PASS only if all five are true. If FAIL, write revised_prompt: a complete "
    "replacement for the generation prompt above (not a suffix), edited to fix every failing "
    "criterion with concrete, visible features (name the parts, their shapes and how they do the "
    "task) while keeping what already works. Fix problems with the SIMPLEST form and FEWEST parts "
    "that work: do not add linkages, gears, screws, sliders or electronics unless the task cannot "
    "be done without them, and prefer removing a problem part to adding a new one. It must still "
    "ask for one entirely novel object, a plain white background, nothing else in the image, no "
    "{exclude}, and no text, numbers, labels, arrows, icons, logos or screens. Do not name or "
    "describe the existing {avoid} it must avoid resembling, beyond saying it must not resemble "
    "them.{history}\n\n"
    "Respond with JSON only: "
    '{{"criteria": {{"evident": {{"ok": true, "reason": "..."}}, "correct": {{"ok": true, "reason": "..."}}, '
    '"specific": {{"ok": true, "reason": "..."}}, "novel": {{"ok": true, "reason": "..."}}, '
    '"no_text": {{"ok": true, "reason": "..."}}}}, '
    '"verdict": "PASS or FAIL", "summary": "one sentence", "revised_prompt": "... or null if PASS"}}'
)
POOL_PROMPT = (
    "This image shows one object on a white background. Which ONE of the following tasks is this "
    "object designed for? Judge only from its shape, parts and materials.\n\n{options}\n\n"
    'Respond with JSON only: {{"choice": <number>}}'
)
POOL_REVIEW = (
    "\n\nPool test: {n_ok} of {n} other vision models, each shown only this image and asked to pick "
    "its task from {k} options (the intended task plus random other tasks from the dataset), chose the "
    "intended task. Their picks: {picks}. "
    "If fewer than {min_ok} chose the intended task, verdict must be FAIL even if the five criteria "
    "hold, and revised_prompt must change the object so it can no longer be mistaken for the tasks "
    "they picked instead ({wrong})."
)
HISTORY_HEADER = ("\n\nEarlier attempts at this same task (learn from them; do not repeat a direction "
                  "that failed or over-correct):\n")


def generate_image(api_key: str, prompt: str):
    """Returns (png_bytes | None, error | None): one text-to-image call to OpenRouter's Image API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/AdaDTur/NVRD",
        "X-Title": "NVRD affordance-guided generation",
    }
    payload = {"model": IMAGE_MODEL, "prompt": prompt, "quality": QUALITY, "aspect_ratio": ASPECT_RATIO}
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_IMAGES_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                print(f"      image call {attempt}/{HTTP_RETRIES} failed: {last_err}", flush=True)
                if resp.status_code in (400, 401, 402, 403):  # not transient (bad request, auth, credits, safety)
                    break
                time.sleep(HTTP_BACKOFF_S * attempt)
                continue
            items = resp.json().get("data") or []
            if not items or not items[0].get("b64_json"):
                last_err = f"no b64_json in response (keys: {list(resp.json().keys())})"
                continue
            return base64.b64decode(items[0]["b64_json"]), None
        except Exception as e:  # noqa: BLE001 -- surfaced to caller, not swallowed
            last_err = str(e)
            print(f"      image call {attempt}/{HTTP_RETRIES} failed: {last_err}", flush=True)
            time.sleep(HTTP_BACKOFF_S * attempt)
    return None, last_err


def _json_from(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    return json.loads(m.group(0))


def ask_judge(client, text: str, img_b64: str) -> dict:
    last = None
    for k in range(1, HTTP_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, max_tokens=4000,  # reasoning model: budget covers hidden reasoning too
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}])
            content = r.choices[0].message.content
            if not content:
                raise RuntimeError(f"empty judge reply (finish_reason={r.choices[0].finish_reason})")
            return _json_from(content)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"      judge call {k}/{HTTP_RETRIES} failed: {e}", flush=True)
            time.sleep(5 * k)
    raise RuntimeError(f"judge failed after {HTTP_RETRIES} tries: {last}")


# Options per labeler = the intended task + N_DISTRACTORS random other tasks (default 14: about half
# of the 31-task pool). With 5 options the tempting wrong task was usually missing and nearly every
# object passed; the full pool flagged real confusions but is long. Half keeps most of that signal.
N_DISTRACTORS = int(os.environ.get("POOL_DISTRACTORS", 14))


def pool_test(client, img_b64: str, pool: list, target: str, seed: int, n_distractors: int = N_DISTRACTORS) -> list:
    """Each labeler picks one task from a short list: the intended task plus n_distractors other
    tasks sampled at random from the pool, shuffled (a fresh draw per labeler, so neither the
    distractors nor the position are a cue). With 15 options, chance is about 7% per labeler.
    Returns [{"model", "choice", "correct", "options"}]."""
    import random
    out = []
    for li, model in enumerate(LABELERS):
        rng = random.Random(seed * 101 + li)
        order = rng.sample([t for t in pool if t != target], min(n_distractors, len(pool) - 1)) + [target]
        rng.shuffle(order)
        options = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(order))
        choice = None
        for k in range(1, HTTP_RETRIES + 1):
            try:
                r = client.chat.completions.create(model=model, max_tokens=600, messages=[{"role": "user", "content": [
                    {"type": "text", "text": POOL_PROMPT.format(options=options)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}])
                n = int(_json_from(r.choices[0].message.content or "")["choice"])
                choice = order[n - 1] if 1 <= n <= len(order) else None
                break
            except Exception as e:  # noqa: BLE001
                print(f"      labeler {model} call {k}/{HTTP_RETRIES} failed: {e}", flush=True)
                time.sleep(5 * k)
        out.append({"model": model, "choice": choice, "correct": choice == target, "options": order})
    return out


def run_one(client, api_key: str, oid: str, task: str, avoid: str, exclude: str, prompt: str, out_dir: str,
            pool: list, review_note: str = "") -> dict:
    obj_dir = os.path.join(out_dir, oid)
    os.makedirs(obj_dir, exist_ok=True)
    rec = {"id": oid, "task": task, "avoid": avoid, "exclude": exclude, "image_model": IMAGE_MODEL,
           "quality": QUALITY, "judge_model": JUDGE_MODEL, "attempts": []}
    history = []
    for n in range(1, MAX_ATTEMPTS + 1):
        png, err = generate_image(api_key, prompt)
        att = {"attempt": n, "prompt": prompt}
        if png is None:
            att["error"] = err
            rec["attempts"].append(att)
            print(f"    attempt {n}: generation FAILED -- {err}", flush=True)
            continue
        # One image per object: every attempt overwrites final.png, so the file left at the end is
        # the final attempt's image. Prompts, picks and scores for every attempt stay in the manifest.
        path = os.path.join(obj_dir, "final.png")
        with open(path, "wb") as f:
            f.write(png)
        img_b64 = base64.b64encode(png).decode("ascii")
        blind = ask_judge(client, BLIND_PROMPT, img_b64)
        labels = pool_test(client, img_b64, pool, task, seed=n)
        n_ok = sum(l["correct"] for l in labels)
        wrong = sorted({l["choice"] for l in labels if not l["correct"] and l["choice"]})
        hist = (HISTORY_HEADER + "\n".join(
            f"  Attempt {h['attempt']}: blind guesses {h['blind'].get('guesses')}; pool picks "
            f"{[l['choice'] for l in h['pool']]}; {h['verdict']} -- {h['summary']}"
            for h in history)) if history else ""
        pool_note = POOL_REVIEW.format(n_ok=n_ok, n=len(labels), k=N_DISTRACTORS + 1, min_ok=MIN_POOL_CORRECT,
                                       picks=json.dumps([l["choice"] for l in labels]),
                                       wrong=", ".join(wrong) or "none")
        review = ask_judge(client, REVIEW_PROMPT.format(
            guesses=json.dumps(blind.get("guesses", [])), task=task, avoid=avoid, exclude=exclude,
            prompt=prompt, history=hist + pool_note + review_note), img_b64)
        passed = (str(review.get("verdict", "FAIL")).upper().startswith("PASS") and n_ok >= MIN_POOL_CORRECT)
        crit = review.get("criteria", {})
        att.update(image=path, blind=blind, pool=labels, pool_correct=n_ok, criteria=crit,
                   verdict="pass" if passed else "fail",
                   summary=review.get("summary"), revised_prompt=review.get("revised_prompt"))
        rec["attempts"].append(att)
        history.append(att)
        failing = [k for k, v in crit.items() if not (isinstance(v, dict) and v.get("ok"))]
        print(f"    attempt {n}: blind={blind.get('guesses')} pool={n_ok}/{len(labels)} {wrong or ''} -> {att['verdict'].upper()}"
              + (f" (failing: {', '.join(failing)})" if failing else "") + f" -- {att['summary']}", flush=True)
        if passed:
            break
        if review.get("revised_prompt"):
            prompt = review["revised_prompt"]
        time.sleep(2)
    done = [a for a in rec["attempts"] if "image" in a]
    if done:  # final.png holds the last attempt's image; the loop stops at the first PASS
        rec.update(final_attempt=done[-1]["attempt"], final_verdict=done[-1]["verdict"])
    return rec


def main():
    from openai import OpenAI
    args = sys.argv[1:]
    affordance_only = "--affordance-only" in args
    primary = "--primary" in args
    args = [a for a in args if a not in ("--affordance-only", "--primary")]
    api_key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY not set (export it or put it in .env)")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    if affordance_only:
        with open(os.path.join(HERE, "synthia_affordances_clustered.txt")) as f:
            vocab = {line.strip() for line in f if line.strip()}
        ids = args or sys.exit("--affordance-only needs one or more verbs from synthia_affordances_clustered.txt")
        bad = [v for v in ids if v not in vocab]
        jobs = {v: (v.replace("_", " "), f"tools used to {v.replace('_', ' ')}", "anything else") for v in ids}
        out_dir = os.path.join(HERE, "figures", "affordance_only_novel_objects")
        json_dir = os.path.join(HERE, "json", "affordance_only_objects")
        template = AFFORDANCE_ONLY_PROMPT
        pool = sorted(vocab)
    else:
        ids = args or [s for s in SCENARIOS if s not in KEEP]
        bad = [s for s in ids if s not in SCENARIOS]
        jobs = {s: SCENARIOS[s] for s in ids if s in SCENARIOS}
        pool = [t for t, _, _ in SCENARIOS.values()]
        if primary:
            ids = [i for i in ids if i in PRIMARY_PARTS] if not args else ids
            bad += [s for s in ids if s in SCENARIOS and s not in PRIMARY_PARTS]
            out_dir = os.path.join(HERE, "figures", "affordance_guided_novel_objects_primary")
            json_dir = os.path.join(HERE, "json", "affordance_guided_objects_primary")
            template = PRIMARY_PROMPT
        else:
            out_dir = os.path.join(HERE, "figures", "affordance_guided_novel_objects_v2")
            json_dir = os.path.join(HERE, "json", "affordance_guided_objects_v2")
            template = SCENARIO_PROMPT
    if bad:
        sys.exit(f"unknown ids: {bad}")

    os.makedirs(json_dir, exist_ok=True)
    mpath = os.path.join(json_dir, "manifest.json")
    manifest = {}
    if os.path.isfile(mpath):
        manifest = {r.get("id", r.get("scenario_id")): r for r in json.load(open(mpath))}
    for oid, (task, avoid, exclude) in jobs.items():
        if manifest.get(oid, {}).get("final_verdict") == "pass":
            print(f"  {oid}: already passed, skipping", flush=True)
            continue
        print(f"\n=== {oid} ({task}) ===", flush=True)
        part = PRIMARY_PARTS.get(oid, "") if (primary and not affordance_only) else ""
        prompt = template.format(task=task, avoid=avoid, exclude=exclude, primary=part)
        note = PRIMARY_REVIEW_NOTE.format(primary=part) if part else ""
        manifest[oid] = run_one(client, api_key, oid, task, avoid, exclude, prompt, out_dir, pool, review_note=note)
        if part:
            manifest[oid]["strategy"] = {"mode": "primary", "primary_part": part}
        with open(mpath, "w") as f:
            json.dump(list(manifest.values()), f, indent=2)
    n_pass = sum(manifest.get(i, {}).get("final_verdict") == "pass" for i in jobs)
    print(f"\n{n_pass}/{len(jobs)} passed -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
