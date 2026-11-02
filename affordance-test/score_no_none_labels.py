"""
Non-destructive companion to relabel_none_cases.py: computes every region's label under the
agd20k_no_none vocabulary (same 36 verbs, no "none") for ALL 30 novel objects, storing it in a new
vocabularies["agd20k_no_none"] key alongside the existing vocabularies["agd20k"] entry -- never
overwriting the original "none" finding. (relabel_none_cases.py was scoped to 5 pilot objects and
intentionally replaced "none" in place, for the live study's model-comparison backend; this script
is for the "what would the model say if forced to commit" supplementary view across the full set.)

Correctness: removing "none" from the candidate set can only change the argmax for a region where
"none" already won (score_options scores every option independently; see relabel_none_cases.py's
docstring for the full argument), so a region whose agd20k label is already a real verb gets
agd20k_no_none = agd20k copied with no GPU work -- only regions currently labeled "none" need
actual re-scoring.

Usage:
    python3 score_no_none_labels.py                 # all 30 novel objects, all 3 models
    python3 score_no_none_labels.py shape_10 shape_14   # restrict to these objects
"""
import json
import os
import sys

import torch

from batch_final_labeling import MODELS
from gpu_contention import wait_for_gpu_headroom
from relabel_none_cases import NO_NONE_VOCAB, VOCAB, out_dir_for, relabel_object

OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
           "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
           "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
           "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
           "shape_39", "shape_40"]
RERUN_MODELS = ["qwen3-vl-8b", "internvl3-8b", "molmo2-8b"]


def copy_unchanged_and_find_none(model_key: str, objects: list) -> dict:
    """For every region NOT labeled "none", copies agd20k -> agd20k_no_none in place (no GPU work)
    and writes the file. Returns {object_name: [region_id, ...]} for the regions that ARE "none"
    and still need real re-scoring."""
    targets = {}
    d = out_dir_for(model_key)
    for obj in objects:
        path = os.path.join(d, f"{obj}_affordance_justification.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            data = json.load(f)
        nones, changed = [], False
        for r in data["regions"]:
            if r["vocabularies"][VOCAB]["label"] == "none":
                nones.append(r["region_id"])
            elif r["vocabularies"].get(NO_NONE_VOCAB) != r["vocabularies"][VOCAB]:
                r["vocabularies"][NO_NONE_VOCAB] = dict(r["vocabularies"][VOCAB])
                changed = True
        if changed:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        if nones:
            targets[obj] = nones
    return targets


def main():
    requested_objects = sys.argv[1:] or OBJECTS

    for model_key in RERUN_MODELS:
        targets = copy_unchanged_and_find_none(model_key, requested_objects)
        if not targets:
            print(f"{model_key}: nothing needs re-scoring in {len(requested_objects)} object(s)")
            continue

        n_regions = sum(len(v) for v in targets.values())
        print(f"\n=== {model_key}: {n_regions} region(s) across {len(targets)} object(s) need re-scoring ===")

        repo_id, model_cls = MODELS[model_key]["repo_id"], MODELS[model_key]["model_cls"]
        trust_remote_code = MODELS[model_key].get("trust_remote_code", False)
        safe_generate = MODELS[model_key].get("safe_generate", False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wait_for_gpu_headroom(log=print)
        print(f"Loading {model_key} ({repo_id})...")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
        model = model_cls.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code
        )
        model.eval()

        from affordance_labeling import VOCABULARIES
        options = VOCABULARIES[NO_NONE_VOCAB]["options"]
        end_token = processor.tokenizer.eos_token
        option_ids = [processor.tokenizer(o + end_token, add_special_tokens=False).input_ids for o in options]

        out_dir = out_dir_for(model_key)
        for obj, region_ids in targets.items():
            wait_for_gpu_headroom(log=print)
            print(f"  {obj}:")
            patched = relabel_object(model, processor, option_ids, device, obj, region_ids, safe_generate)

            path = os.path.join(out_dir, f"{obj}_affordance_justification.json")
            with open(path) as f:
                data = json.load(f)
            for r in data["regions"]:
                if r["region_id"] in patched:
                    r["vocabularies"][NO_NONE_VOCAB] = patched[r["region_id"]]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  saved {path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
