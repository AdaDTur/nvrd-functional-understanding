"""
Per-part functionality banks for the human study: the cumulative top-10 across the evaluated
models' answers on the synthia vocabulary (Phi-3.5 excluded, see MODELS).

Sources: json/affordance_justification_<model>_synthia/<obj>_affordance_justification.json for
molmo2-4b, gemma4-e4b, qwen3.5-4b (batch_final_labeling.py, AFF_VOCAB=synthia_filtered) and gpt-5.4
(label_gpt_api.py --rank-only), with the generation prompt's option list shuffled per part
(SHUFFLE_OPTIONS=1) so answers aren't pulled toward the list's first words. Scoring uses each model's
GENERATION top-10 (vocabularies.synthia_filtered.generation.ranked), the one mode all four share: rank 1 = 10 points ...
rank 10 = 1, summed over models; ties broken by how many models listed it, then alphabetically.
Off-vocabulary answers were already dropped, so some lists are short; if fewer than 10 options get
any points, the remaining slots are filled the same way from the open models' log-prob top-10s.

Written into the study manifests (prolific-novel-concepts/affordance-study/manifests/shape_N.json):
  region["bank"]           that part's 10 options, alphabetised (display order doesn't reveal the ranking)
  manifest["overall_bank"] whole-object question: the same scoring over every part's lists
  manifest["bank_source"]  provenance

Caveat: options drawn from the models pull human answers toward the models' answers, inflating
human-model agreement; free typing is still allowed.

Usage (from nvrd/affordance-test/): python3 code/study_assets/build_model_banks.py
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # affordance-test
STUDY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "prolific-novel-concepts", "affordance-study", "manifests")
# Phi-3.5 left out: on the alphabetical synthia list 96% of its generated answers came from the first
# 20 words, i.e. it copied the list rather than answering (2026-11-18).
MODELS = ["molmo2-4b", "gemma4-e4b", "qwen3.5-4b", "gpt-5.4"]
VOCAB = "synthia_filtered"
K = 10


def load(obj: str):
    """region_id -> (generation lists over all models, log-prob lists over the open models)"""
    gen, lp = collections.defaultdict(list), collections.defaultdict(list)
    for m in MODELS:
        with open(os.path.join(HERE, "json", f"affordance_justification_{m}_synthia", f"{obj}_affordance_justification.json")) as f:
            for r in json.load(f)["regions"]:
                v = r["vocabularies"][VOCAB]
                gen[r["region_id"]].append(v["generation"]["ranked"])
                if "logprob" in v:
                    lp[r["region_id"]].append([x["option"] for x in v["logprob"]["ranked"]])
    return gen, lp


def ranked(lists: list) -> list:
    score, count = collections.Counter(), collections.Counter()
    for lst in lists:
        for i, opt in enumerate(lst[:K]):
            score[opt] += K - i
            count[opt] += 1
    return sorted(score, key=lambda o: (-score[o], -count[o], o))


def bank(gen_lists: list, lp_lists: list) -> tuple:
    top = ranked(gen_lists)[:K]
    n_filled = 0
    for o in ranked(lp_lists):
        if len(top) >= K:
            break
        if o not in top:
            top.append(o)
            n_filled += 1
    return sorted(top), n_filled


def main():
    n_obj = n_reg = n_fill = 0
    for fn in sorted(os.listdir(STUDY)):
        if not fn.startswith("shape_"):
            continue
        obj = fn[:-5]
        gen, lp = load(obj)
        p = os.path.join(STUDY, fn)
        man = json.load(open(p))
        for r in man["regions"]:
            r["bank"], filled = bank(gen[r["region_id"]], lp[r["region_id"]])
            n_fill += filled > 0
            n_reg += 1
        man["overall_bank"], _ = bank([l for rid in gen for l in gen[rid]], [l for rid in lp for l in lp[rid]])
        man["bank_source"] = {
            "models": MODELS, "vocabulary": VOCAB, "mode": "generation top-10 (option order shuffled per part), Phi-3.5 excluded",
            "scoring": "Borda (rank 1 = 10 ... rank 10 = 1) summed over models, top 10, shown alphabetically",
            "fill": "open models' log-prob top-10s, only when fewer than 10 options scored",
            "overall_bank": "same scoring over every part's lists",
        }
        json.dump(man, open(p, "w"), indent=2)
        n_obj += 1
    print(f"banks written for {n_reg} regions in {n_obj} manifests -> {STUDY}; {n_fill} parts needed log-prob fill")


if __name__ == "__main__":
    main()
