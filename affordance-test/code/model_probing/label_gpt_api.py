"""
GPT-5.4 counterpart to batch_final_labeling.py: same parts (compute_parts), same numbered overlay
(numbered_overlay + downscale to MAX_SIDE), same per-part forced-choice question and same
justification prompt, same output schema -- so its JSONs drop into everything that reads
json/affordance_justification_<model_key>/.

What differs, and why: the open models are scored by teacher-forced log P(option) over every option
(score_options). OpenRouter's openai/gpt-5.4 exposes no logprobs, so instead the forced choice is a
structured-output call whose JSON schema constrains "label" to an enum of exactly the vocabulary's
options -- still a guaranteed single valid option, but a sampled argmax-ish pick, not a full
distribution. Temperature isn't settable on GPT-5 reasoning models; reasoning effort defaults to
"none" so it isn't given a deliberation budget the open models don't get. Note this asymmetry when
comparing GPT-5.4 against the open models.

Top-10: GPT gets only the direct-generation mode (no logprobs, so no log-prob mode), stored as
vocabularies[<vocab>]["generation"] = {"ranked": [up to 10 options], "n_valid", "invalid", "raw"},
the same free-text prompt, parser and fields as the open models' generation mode in
batch_final_labeling.py (no JSON schema here, so format-following counts equally for everyone). That is
one extra API call per part. The "label" above still comes from its own single-answer call, so
existing results stay as they were.

Routed through OpenRouter (OPENROUTER_API_KEY in affordance-test/.env), same as the image-generation
scripts, since this project's own OpenAI org hit its spend limit.

Usage:
    venv_vlm/bin/python3 label_gpt_api.py --split novel            # the 30 original novel objects
    venv_vlm/bin/python3 label_gpt_api.py shape_16 shape_27        # specific objects
    venv_vlm/bin/python3 label_gpt_api.py --split novel --resume   # skip objects already done
    AFF_VOCAB=synthia_filtered venv_vlm/bin/python3 label_gpt_api.py --split novel --rank-only
        # top-10 only, on the synthia vocabulary -> affordance_justification_gpt-5.4_synthia/
    venv_vlm/bin/python3 label_gpt_api.py --split novel --generation-only  # redo only the ranked
        # lists of existing outputs that predate the free-text format, keeping their labels
Env: GPT_MODEL (default openai/gpt-5.4), GPT_REASONING_EFFORT (default none), GPT_WORKERS (default 8).
"""
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test

from PIL import Image
from pycocotools import mask as mask_utils

from affordance_labeling import (
    JSON_DIR, TOP_K, VOCABULARIES, build_per_part_question, build_ranked_question, downscale, first_sentence,
    numbered_overlay, parse_ranked, shuffled_options,
)
from batch_final_labeling import (
    OBJECTS, REGION_SOURCE, VOCAB, build_per_part_justification_prompt, compute_parts, has_topk_fields,
)

# AFF_VOCAB=synthia_filtered runs on the 230-verb synthia list instead; outputs then go to
# affordance_justification_<model>_<vocab>/ so the AGD20K results are never overwritten.
VOCAB = os.environ.get("AFF_VOCAB", VOCAB)
# SHUFFLE_OPTIONS=1: per-part shuffled option order in the top-K prompt (see shuffled_options); the
# same seed (region_id) as batch_final_labeling.py, so every model sees the same order for a part.
SHUFFLE_OPTIONS = os.environ.get("SHUFFLE_OPTIONS") == "1"

MODEL = os.environ.get("GPT_MODEL", "openai/gpt-5.4")
MODEL_KEY = MODEL.split("/")[-1]  # "gpt-5.4"
REASONING_EFFORT = os.environ.get("GPT_REASONING_EFFORT", "none")
WORKERS = int(os.environ.get("GPT_WORKERS", "8"))
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 5


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY not set and not in affordance-test/.env")


def image_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(api_key: str, image_uri: str, text: str, response_format: "dict | None" = None,
         max_tokens: int = 200) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": text},
        ]}],
        "reasoning": {"effort": REASONING_EFFORT},
        "max_tokens": max_tokens,
        "provider": {"require_parameters": True},  # fail loudly rather than silently drop the schema
    }
    if response_format:
        body["response_format"] = response_format
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=120)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content")
                if content:
                    return content
                last_err = f"empty content: {r.text[:300]}"
            else:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403):  # not transient
                    break
        except requests.RequestException as e:
            last_err = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{MODEL} call failed after retries: {last_err}")


def label_part(api_key, image_uri, i, n_parts, options, glosses) -> tuple:
    schema = {"type": "json_schema", "json_schema": {"name": "affordance", "strict": True, "schema": {
        "type": "object", "properties": {"label": {"type": "string", "enum": options}},
        "required": ["label"], "additionalProperties": False,
    }}}
    raw = chat(api_key, image_uri, build_per_part_question(i, n_parts, options, glosses), schema)
    label = json.loads(raw)["label"]
    assert label in options, f"schema-violating label {label!r}"
    just_raw = chat(api_key, image_uri, build_per_part_justification_prompt(i, n_parts, label, glosses))
    # The open models get "<label>:" force-prefixed onto their answer; GPT-5.4 sometimes echoes it back.
    just = just_raw.strip()
    if just.lower().startswith(f"{label}:"):
        just = just[len(label) + 1:].strip()
    ranked = rank_part(api_key, image_uri, i, n_parts, options, glosses)
    return label, first_sentence(just), ranked


def rank_part(api_key, image_uri, i, n_parts, options, glosses, region_id: str = "") -> dict:
    """Direct-generation top-K, on exactly the same footing as the open models'
    (batch_final_labeling.generate_ranked): the same build_ranked_question prompt asking for a
    numbered plain-text list, parsed by the same parse_ranked. Deliberately NOT a JSON schema with
    an enum of the vocabulary -- that constrained GPT to valid, well-formed options while the open
    models' free text could drift off-vocabulary or repeat, so n_valid compared format compliance
    rather than the ranking itself."""
    listed = shuffled_options(options, region_id) if (SHUFFLE_OPTIONS and region_id) else options
    raw = chat(api_key, image_uri, build_ranked_question(i, n_parts, listed, glosses), max_tokens=600)
    return {**parse_ranked(raw, options), "raw": raw.strip()}


def regenerate_generation_only(api_key: str, out_path: str) -> dict:
    """--generation-only: redo just the ranked-list call for an existing output, keeping its
    label/justification (re-running those would resample GPT's answers, not just the ranking).
    Parts are recomputed with compute_parts and must match the saved ones exactly."""
    with open(out_path) as f:
        data = json.load(f)
    parts, img_path = compute_parts(data["object"])
    assert [p["region_id"] for p in parts] == [r["region_id"] for r in data["regions"]], \
        f"{data['object']}: recomputed parts differ from the saved output"
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image_uri = image_data_uri(downscale(numbered_overlay(Image.open(img_path).convert("RGB"), masks)))
    options, glosses = VOCABULARIES[VOCAB]["options"], VOCABULARIES[VOCAB]["glosses"]
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(rank_part, api_key, image_uri, i, len(parts), options, glosses)
                   for i in range(1, len(parts) + 1)]
        gens = [f.result() for f in futures]
    for r, g in zip(data["regions"], gens):
        r["vocabularies"][VOCAB]["generation"] = g
    data["generation_format"] = "free_text_numbered_list"
    return data


def process_object(api_key: str, object_name: str, rank_only: bool = False) -> dict:
    parts, img_path = compute_parts(object_name)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    overlay = downscale(numbered_overlay(Image.open(img_path).convert("RGB"), masks))
    image_uri = image_data_uri(overlay)
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]

    if rank_only:  # --rank-only: just the top-K generation call (one call per part), no label/justification
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(rank_part, api_key, image_uri, i, len(parts), options, glosses,
                                   parts[i - 1]["region_id"])
                       for i in range(1, len(parts) + 1)]
            gens = [f.result() for f in futures]
        for part, g in zip(parts, gens):
            print(f"    {part['region_id']:>28s}  {g['ranked']}", flush=True)
        return {
            "object": object_name, "model": MODEL, "image_path": img_path, "region_source": REGION_SOURCE,
            "method": "rank_only", "reasoning_effort": REASONING_EFFORT,
            "generation_option_order": "shuffled_per_part" if SHUFFLE_OPTIONS else "vocabulary", "top_k": TOP_K, "modes": ["generation"],
            "generation_format": "free_text_numbered_list",
            "regions": [{**p, "vocabularies": {VOCAB: {"generation": g}}} for p, g in zip(parts, gens)],
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(label_part, api_key, image_uri, i, len(parts), options, glosses)
                   for i in range(1, len(parts) + 1)]
        answers = [f.result() for f in futures]

    results = []
    for part, (label, justification, generation) in zip(parts, answers):
        # No "logprob" entry: the API exposes no logprobs, so GPT only has the generation mode.
        results.append({**part, "vocabularies": {VOCAB: {
            "label": label, "justification": justification, "generation": generation,
        }}})
        print(f"    {part['region_id']:>28s}  {label:>14s}: {justification}  "
              f"[ranked {generation['n_valid']}/{TOP_K}]", flush=True)
    return {
        "object": object_name, "model": MODEL, "image_path": img_path, "region_source": REGION_SOURCE,
        "method": "per_part_with_comparative_context_structured_output",
        "reasoning_effort": REASONING_EFFORT, "top_k": TOP_K, "modes": ["generation"],
        "generation_format": "free_text_numbered_list", "regions": results,
    }


def main():
    args = sys.argv[1:]
    resume = "--resume" in args
    generation_only = "--generation-only" in args
    rank_only = "--rank-only" in args
    args = [a for a in args if a not in ("--resume", "--generation-only", "--rank-only")]
    split = None
    if args and args[0] == "--split":
        split, args = args[1], args[2:]
    suffix = "" if VOCAB == "agd20k" else f"_{VOCAB.split('_')[0]}"
    out_dir = os.path.join(JSON_DIR, f"affordance_justification_{MODEL_KEY}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    todo = [
        name for category, names in OBJECTS for name in names
        if (not args or name in args) and (split is None or category == split)
        and not (resume and has_topk_fields(os.path.join(out_dir, f"{name}_affordance_justification.json"),
                                            VOCAB, modes=("generation",)))
    ]
    if generation_only:  # only existing outputs still holding the old schema-constrained ranking
        def _needs(name):
            path = os.path.join(out_dir, f"{name}_affordance_justification.json")
            if not os.path.isfile(path):
                return False
            with open(path) as f:
                return json.load(f).get("generation_format") != "free_text_numbered_list"
        todo = [n for category, names in OBJECTS for n in names
                if (not args or n in args) and (split is None or category == split) and _needs(n)]
    api_key = load_api_key()
    print(f"{MODEL} (reasoning effort {REASONING_EFFORT}){' [generation only]' if generation_only else ''}: "
          f"{len(todo)} objects -> {out_dir}", flush=True)
    for name in todo:
        print(f"  {name}:", flush=True)
        out_path = os.path.join(out_dir, f"{name}_affordance_justification.json")
        try:
            data = (regenerate_generation_only(api_key, out_path) if generation_only
                    else process_object(api_key, name, rank_only))
        except Exception as e:
            print(f"  {name}: FAILED -- {e}", flush=True)
            continue
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_distinct = len({r["vocabularies"][VOCAB].get("label") for r in data["regions"]})
        print(f"  saved {out_path}  (distinct labels: {n_distinct} of {len(data['regions'])} parts)", flush=True)


if __name__ == "__main__":
    main()
