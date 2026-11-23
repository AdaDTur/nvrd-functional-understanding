"""
Coarse functional groups over the synthia study vocabulary, for "agreement when similar functionalities
are clustered" (compare_human_model.py). The vocabulary (synthia_affordances_clustered.txt, 230 labels +
"none") is already synonym-merged; this groups it one level coarser into broad functions (e.g. support /
stand / balance / rest), and maps people's free-typed answers onto those groups.

One LLM call over every label at once (same reasoning as cluster_synthia_affordances.py: groups span the
whole list), via OpenRouter. Review and hand-edit the output as needed; the analysis reads it as is.

Usage (from nvrd/affordance-test/): OPENROUTER_API_KEY=... python3 code/vocab_curation/group_synthia_functions.py
Output: json/synthia_function_groups.json  {"groups": {name: [labels]}, "typed": {answer: [group names]}}
"""
import csv
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
MODEL = os.environ.get("GROUP_MODEL", "openai/gpt-5.4")
OUT = os.path.join(HERE, "json", "synthia_function_groups.json")


def main():
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    vocab = [l.strip() for l in open(os.path.join(HERE, "synthia_affordances_clustered.txt")) if l.strip()]
    vocab_set = {v.lower() for v in vocab} | {"none"}
    typed = set()
    p = os.path.join(HERE, "responses_long.csv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p, newline="")):
            for x in r["selected"].split("|"):
                x = " ".join(x.strip().lower().split())
                if x and x not in vocab_set:
                    typed.add(x)
    prompt = (
        "Below is a vocabulary of object functionalities (what an object or one of its parts can be used to do). "
        "Group ALL of them into 20 to 30 broad functional groups, so that labels a person would treat as roughly "
        "the same use of an object end up together (for example support / stand / balance / rest / prop; or "
        "attach / anchor / clamp / fasten / lock). Every label must appear in exactly one group; keep 'none' in "
        "its own group called 'none'. Name each group with a short, plain word or phrase.\n\n"
        "Then map each free-typed answer (written by study participants; some list several uses or contain "
        "comments) to the group or groups it expresses, or to [] if it states no functionality.\n\n"
        f"VOCABULARY ({len(vocab) + 1}): " + json.dumps(vocab + ["none"]) + "\n\n"
        f"TYPED ANSWERS ({len(typed)}): " + json.dumps(sorted(typed)) + "\n\n"
        'Respond with JSON only: {"groups": {"<group name>": ["label", ...], ...}, '
        '"typed": {"<typed answer>": ["<group name>", ...], ...}}'
    )
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=600,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"}, "max_tokens": 16000})
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    labels = [l for g in data["groups"].values() for l in g]
    missing = sorted(vocab_set - {l.lower() for l in labels})
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    data["model"] = MODEL
    json.dump(data, open(OUT, "w"), indent=2)
    print(f"{len(data['groups'])} groups -> {OUT}; missing labels: {missing}; duplicated: {dupes}")
    for g, ls in data["groups"].items():
        print(f"  {g} ({len(ls)}): {', '.join(ls[:12])}{' ...' if len(ls) > 12 else ''}")


if __name__ == "__main__":
    main()
