"""
LLM relevance filter, step 1 of 3 for cutting the SYNTHIA affordance ontology (661 unique terms,
see synthia_affordances.txt / score_no_none_labels.py's sibling investigation) down to a usable
replacement/extension for the current AGD20K-derived study vocabulary (36 verbs + "none").

SYNTHIA's ontology spans far more than single-object-part manipulation (it has a "geography"
supercategory, vehicles, software/social concepts, etc.), so most of the list is out of domain for
this study rather than merely redundant -- a pure embedding-similarity/dedup pass can't tell
"predict" or "decelerate" don't apply, only that they're dissimilar from other terms. This script
does the relevance pass with an LLM (gemini-3.8-flash: fast, text-only here so no vision tower
cost, via the same google-genai integration already proven in this project -- see
generate_human_asset.py); step 2
(embedding-based synonym clustering on the survivors) and step 3 (manual eyeball of the final
~60-100 candidates) are separate, later passes, not done here.

Terms are sent in batches (not one giant single prompt) so the model's answers stay auditable per
batch and a bad batch is cheap to re-run; each term gets a one-line reason recorded alongside its
verdict specifically so the yes/no calls can be spot-checked by hand, not just trusted blindly.

Usage:
    venv/bin/python3 filter_synthia_affordances.py
Output:
    json/synthia_affordance_filter.json      -- full {term, applicable, reason} list
    synthia_affordances_filtered.txt         -- just the "applicable" terms, one per line, sorted
"""
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
_ENV_FILE = os.path.join(HERE, ".env")
if os.path.isfile(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

IN_PATH = os.path.join(HERE, "synthia_affordances.txt")
OUT_JSON = os.path.join(HERE, "json", "synthia_affordance_filter.json")
OUT_TXT = os.path.join(HERE, "synthia_affordances_filtered.txt")
MODEL = "gemini-3.8-flash"
BATCH_SIZE = 40

INSTRUCTIONS = """You are helping curate a vocabulary of FUNCTIONAL AFFORDANCES for labeling parts \
of small, physical, graspable objects (tools, household items, novel sculpted shapes) from \
photographs -- the same kind of task as "what could this handle be used to do? -> grip, pull".

IMPORTANT -- an affordance is what the part LETS A PERSON DO WITH IT, not an action the part \
itself performs. Judge each term by: "could a person use some part of some small physical object \
to {term}?" -- NOT "does the part itself {term}". Under this reading "ride" is applicable (a \
bike seat/saddle affords riding), "kick" is applicable (a ball or pedal affords kicking), "jump" \
is applicable (a trampoline surface or springboard affords jumping) -- all are things a person \
DOES USING the part, exactly like "cut" (a blade affords cutting), "pour" (a spout affords \
pouring), "wear" (a strap affords wearing), "sit on" (a seat affords sitting), "scoop" (a bowl \
affords scooping), "carry" (a handle affords carrying). Don't reject a term just because the verb \
sounds like it describes a person's own body/motion -- what matters is whether some object part \
could plausibly be the thing enabling that action.

For each term below, decide whether it describes a plausible part-affordance of this kind. Mark \
a term NOT applicable if it is scene/environment-level (e.g. "border", "demarcate"), a \
whole-vehicle/traffic-system action no single part enables ("decelerate", "pull over", "air \
traffic control"), software/computational ("predict", "authenticate"), purely social/abstract/\
financial with no physical part involved ("command", "charge" as in a toll, "negotiate"), or \
otherwise not something tied to physically using a specific part of a specific object. \
Near-duplicate or oddly-specific physical terms (e.g. a very narrow synonym of a more common one) \
should still be marked applicable here -- deduplication is a separate later step, this pass is \
ONLY about physical relevance.

Respond with a JSON array, one object per term in the SAME ORDER given, each with exactly these \
keys: "term" (copy exactly), "applicable" (true/false), "reason" (under 10 words).

Terms:
{terms}"""


def classify_batch(client, types, terms: list) -> list:
    # Not str.format(): the instructions text itself contains literal "{term}" placeholders
    # (describing HOW to judge each term), which .format() would misread as missing fields.
    prompt = INSTRUCTIONS.replace("{terms}", "\n".join(f"- {t}" for t in terms))
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "term": {"type": "STRING"},
                "applicable": {"type": "BOOLEAN"},
                "reason": {"type": "STRING"},
            },
            "required": ["term", "applicable", "reason"],
        },
    }
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        ),
    )
    results = json.loads(response.text)
    returned = {r["term"] for r in results}
    missing = set(terms) - returned
    if missing:
        raise ValueError(f"model dropped {len(missing)} term(s) from this batch: {sorted(missing)}")
    return results


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set -- add it to .env and rerun.")
        sys.exit(1)
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    with open(IN_PATH) as f:
        terms = [line.strip() for line in f if line.strip()]
    print(f"{len(terms)} terms loaded from {IN_PATH}")

    all_results = []
    for i in range(0, len(terms), BATCH_SIZE):
        batch = terms[i:i + BATCH_SIZE]
        print(f"batch {i // BATCH_SIZE + 1}/{-(-len(terms) // BATCH_SIZE)} ({len(batch)} terms)...")
        for attempt in range(3):
            try:
                all_results.extend(classify_batch(client, types, batch))
                break
            except Exception as e:
                print(f"  retry {attempt + 1}/3 after error: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"batch starting at {i} failed 3 times")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    applicable = sorted({r["term"] for r in all_results if r["applicable"]})
    with open(OUT_TXT, "w") as f:
        for t in applicable:
            f.write(t + "\n")

    n_yes, n_no = len(applicable), len(all_results) - len(applicable)
    print(f"\n{n_yes} applicable, {n_no} filtered out ({n_yes}/{len(all_results)} kept)")
    print(f"full results -> {OUT_JSON}")
    print(f"filtered list -> {OUT_TXT}")


if __name__ == "__main__":
    main()
