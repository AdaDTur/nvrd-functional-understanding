"""
Targeted re-scoring for parts that came back "none" under the standard forced-choice vocabulary
(VOCABULARIES["agd20k"] in affordance_labeling.py): re-runs just those (model, object, region_id)
triples against VOCABULARIES["agd20k_no_none"] (the same 36 verbs, minus "none"), forcing the model
to commit to a real answer instead of abstaining.

Patches only the touched regions in place in each model's existing
json/affordance_justification_<model>/<object>_affordance_justification.json -- every other region's
label/justification is left untouched. Mirrors batch_final_labeling.py's process_object() scoring
logic exactly (same overlay, same per-part prompt shape, same score_options call) so the only
difference from the original run is the options list.

Correctness note on scoping to just the "none" regions rather than redoing every part of a touched
object: score_options computes every option's log-likelihood independently per part, so removing
"none" from the candidate set can only change the argmax for a part where "none" was already the
winner. Parts that already had a real-word label are provably unaffected and don't need re-scoring.

Usage:
    python3 relabel_none_cases.py                      # scan SCOPE_OBJECTS, all 3 models
    python3 relabel_none_cases.py shape_10 shape_14     # restrict to these objects
"""
import json
import os
import sys

import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModelForImageTextToText, AutoProcessor

from affordance_labeling import VOCABULARIES, build_per_part_question, build_prompt_text, downscale, numbered_overlay, score_options
from batch_final_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, JSON_DIR, MODELS, build_per_part_justification_prompt, compute_parts,
    generate_justification, top1_label,
)
from gpu_contention import wait_for_gpu_headroom

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
VOCAB = "agd20k"
NO_NONE_VOCAB = "agd20k_no_none"

# The 5 objects behind the pilot_2 results export this re-run was asked for; pass object names on
# the command line to cover others instead.
SCOPE_OBJECTS = ["shape_10", "shape_14", "shape_26", "shape_27", "shape_38"]

RERUN_MODELS = ["qwen3-vl-8b", "internvl3-8b", "molmo2-8b"]  # qwen2-vl-7b wasn't run for the study


def out_dir_for(model_key: str) -> str:
    return AFFORDANCE_JUSTIFICATION_DIR if model_key == "qwen2-vl-7b" else os.path.join(
        JSON_DIR, f"affordance_justification_{model_key}"
    )


def find_none_targets(model_key: str, objects: list) -> dict:
    """{object_name: [region_id, ...]} for every region currently labeled "none" by this model."""
    targets = {}
    d = out_dir_for(model_key)
    for obj in objects:
        path = os.path.join(d, f"{obj}_affordance_justification.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        nones = [r["region_id"] for r in data["regions"] if r["vocabularies"][VOCAB]["label"] == "none"]
        if nones:
            targets[obj] = nones
    return targets


def relabel_object(model, processor, option_ids, device, object_name: str, region_ids: list, safe_generate: bool):
    """Recomputes this object's parts/overlay (deterministic, same as the original run), re-scores
    only the requested region_ids against the no-none vocabulary, and returns the patched
    {region_id: {label, justification}} map."""
    parts, img_path = compute_parts(object_name)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(img_path).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    options = VOCABULARIES[NO_NONE_VOCAB]["options"]
    glosses = VOCABULARIES[NO_NONE_VOCAB]["glosses"]

    wanted = set(region_ids)
    patched = {}
    for i, part in enumerate(parts, start=1):
        if part["region_id"] not in wanted:
            continue
        fc_prompt = build_prompt_text(processor, build_per_part_question(i, len(parts), options, glosses))
        fc = score_options(model, processor, overlay, fc_prompt, option_ids, device)
        label = top1_label(dict(zip(options, fc.tolist())))

        just_prompt = build_prompt_text(
            processor, build_per_part_justification_prompt(i, len(parts), label, glosses)
        )
        justification = generate_justification(
            model, processor, overlay, just_prompt, label, device, safe_generate=safe_generate,
        )
        patched[part["region_id"]] = {"label": label, "justification": justification}
        print(f"    {part['region_id']:>28s}  none -> {label:>14s}: {justification}")

    missing = wanted - set(patched)
    if missing:
        print(f"    WARNING: region_id(s) not found in recomputed parts for {object_name}: {sorted(missing)}")
    return patched


def main():
    requested_objects = sys.argv[1:] or SCOPE_OBJECTS

    for model_key in RERUN_MODELS:
        targets = find_none_targets(model_key, requested_objects)
        if not targets:
            print(f"{model_key}: no 'none' cases in {requested_objects}, skipping")
            continue

        n_regions = sum(len(v) for v in targets.values())
        print(f"\n=== {model_key}: {n_regions} region(s) across {len(targets)} object(s) ===")
        for obj, rids in targets.items():
            print(f"  {obj}: {rids}")

        repo_id, model_cls = MODELS[model_key]["repo_id"], MODELS[model_key]["model_cls"]
        trust_remote_code = MODELS[model_key].get("trust_remote_code", False)
        safe_generate = MODELS[model_key].get("safe_generate", False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wait_for_gpu_headroom(log=print)
        print(f"Loading {model_key} ({repo_id})...")
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
        model = model_cls.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code
        )
        model.eval()

        options = VOCABULARIES[NO_NONE_VOCAB]["options"]
        end_token = processor.tokenizer.eos_token
        option_ids = [processor.tokenizer(o + end_token, add_special_tokens=False).input_ids for o in options]

        out_dir = out_dir_for(model_key)
        for obj, region_ids in targets.items():
            wait_for_gpu_headroom(log=print)  # re-checked before every object, not just once at
                                               # the top -- this is what catches a contending job
                                               # that shows up mid-run, which the one-time check
                                               # before the whole script starts cannot
            print(f"  {obj}:")
            patched = relabel_object(model, processor, option_ids, device, obj, region_ids, safe_generate)

            path = os.path.join(out_dir, f"{obj}_affordance_justification.json")
            with open(path) as f:
                data = json.load(f)
            for r in data["regions"]:
                if r["region_id"] in patched:
                    r["vocabularies"][VOCAB] = patched[r["region_id"]]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  saved {path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
