"""
Step 2 of 3 for cutting SYNTHIA down to a usable study vocabulary (see
filter_synthia_affordances.py for step 1, the relevance filter -- this reads its output,
synthia_affordances_filtered.txt, 473 terms after that pass).

This is a redundancy/dedup pass, not a relevance pass: groups near-synonymous terms (e.g.
grasp/grip/clutch/clench) under one canonical label, the same "one label + accepted synonyms"
pattern already used elsewhere in this study (see TRAINING_ITEMS' "accepted" lists in
affordance-study/demo.js, and VOCABULARIES' "glosses" dict in affordance_labeling.py).

Done as ONE LLM call over all 473 terms together (not batched like the relevance pass), on purpose:
synonym relationships can span arbitrarily across the list, and batching would fragment clusters
that happen to land in different batches (e.g. "grip" in batch 3, "grasp" in batch 9 would never
get grouped). The relevance pass didn't have this problem since each term there was judged
independently of the others. Embedding-similarity clustering (cosine distance + a threshold) was
the original plan but was dropped in favor of this: picking a good canonical label per cluster and
matching this project's existing short, common-word AGD20K-style phrasing needs the same semantic
judgment an LLM is already doing here, and a pure-distance threshold can't make that call (how
close is "close enough" varies cluster to cluster, e.g. grasp/grip should merge but hold/carry
arguably shouldn't even though they're semantically related).

Usage:
    venv/bin/python3 cluster_synthia_affordances.py
Output:
    json/synthia_affordance_clusters.json   -- [{canonical, members: [...]}], one per cluster
    synthia_affordances_clustered.txt       -- just the canonical labels, one per line
"""
import json
import os
import sys

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

IN_PATH = os.path.join(HERE, "synthia_affordances_filtered.txt")
OUT_JSON = os.path.join(HERE, "json", "synthia_affordance_clusters.json")
OUT_TXT = os.path.join(HERE, "synthia_affordances_clustered.txt")
MODEL = "gemini-3.8-flash"

INSTRUCTIONS = """You are curating a FORCED-CHOICE vocabulary for labeling the functional \
affordance of parts of small physical objects from photographs (the labeler must pick exactly \
one word/short phrase per part, like AGD20K's style: "cut", "pour", "carry", "sit on").

Below are 473 terms that already passed a relevance filter (each one IS a plausible physical \
part-affordance). Many are near-synonyms of each other for this purpose -- e.g. "grasp", "grip", \
"clutch", and "clench" would all be picked for the same physical situation (something graspable), \
so a labeler forced to choose between all 4 as separate options would be needlessly confused about \
which one is "more correct". Your job: GROUP every term into clusters of near-synonyms (terms \
that, for the purpose of labeling what a part affords, a labeler could not reliably tell apart), \
and pick ONE canonical label per cluster -- the shortest, plainest, most common everyday word or \
short phrase for that action (prefer existing AGD20K-style wording when a cluster matches one of: \
beat, carry, catch, cut, drag, eat, hit, hold, jump, kick, lift, open, pack, peel, pour, push, \
ride, stick, stir, swing, throw, wash, wear, write).

Be conservative about merging: only group terms that are TRULY interchangeable for this task, not \
just related. "hold" and "carry" are both plausible but distinct (carrying implies moving it, \
holding doesn't) -- keep separate. "grasp"/"grip"/"clutch" ARE interchangeable here -- merge. A \
term with no real synonym in the list is its own cluster of 1 (canonical = itself).

EVERY one of the 473 terms must appear in EXACTLY ONE cluster's "members" list (including the \
canonical term itself, listed as a member of its own cluster). Do not drop or invent terms.

Respond with a JSON array of clusters, each: {"canonical": "...", "members": ["...", ...]}.

Terms:
{terms}"""


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

    prompt = INSTRUCTIONS.replace("{terms}", "\n".join(f"- {t}" for t in terms))
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "canonical": {"type": "STRING"},
                "members": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["canonical", "members"],
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
    clusters = json.loads(response.text)

    # Validate full coverage: every input term appears in exactly one cluster. Any term the model
    # dropped becomes its own singleton cluster rather than silently vanishing from the vocabulary;
    # any term the model duplicated across clusters is kept only in the first cluster it appears in.
    seen = set()
    cleaned = []
    for c in clusters:
        members = [m for m in c["members"] if m not in seen]
        seen.update(members)
        if members:
            canonical = c["canonical"] if c["canonical"] in members else members[0]
            cleaned.append({"canonical": canonical, "members": members})

    missing = [t for t in terms if t not in seen]
    for t in missing:
        cleaned.append({"canonical": t, "members": [t]})
    if missing:
        print(f"NOTE: model dropped {len(missing)} term(s), added as singleton clusters: {missing}")

    input_set, covered_set = set(terms), {m for c in cleaned for m in c["members"]}
    extra = covered_set - input_set
    if extra:
        print(f"WARNING: model invented {len(extra)} term(s) not in the input, dropping: {sorted(extra)}")
        for c in cleaned:
            c["members"] = [m for m in c["members"] if m in input_set]
        cleaned = [c for c in cleaned if c["members"]]

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(cleaned, f, indent=2)
    with open(OUT_TXT, "w") as f:
        for c in sorted(cleaned, key=lambda c: c["canonical"]):
            f.write(c["canonical"] + "\n")

    n_multi = sum(1 for c in cleaned if len(c["members"]) > 1)
    print(f"\n{len(terms)} terms -> {len(cleaned)} canonical labels "
          f"({n_multi} clusters merged >1 term, {len(cleaned) - n_multi} singletons)")
    print(f"full clusters -> {OUT_JSON}")
    print(f"canonical labels -> {OUT_TXT}")


if __name__ == "__main__":
    main()
