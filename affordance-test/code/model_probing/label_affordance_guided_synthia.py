"""
Labels the 5 affordance-guided novel objects (figures/affordance_guided_novel_objects/base_images/
<scenario_id>.png, SAM3 regions at json/affordance_guided_objects_regions/<scenario_id>.json) with
the same synthia_filtered vocabulary and per-part + whole-object question shape label_synthia_vocab
.py already uses for the main 30 shape_N novel objects -- so the two sets are directly comparable.

Model: qwen3-vl-8b, same default as label_synthia_vocab.py -- and, independently, confirmed to be
the right pick here too: a fresh count across the main 30 objects' existing agd20k-vocabulary labels
found Qwen3-VL-8B said "none" on 59.1% of parts (vs. InternVL3-8B 3.7%, Molmo2-8B 39.5%), by far the
most of the 3 models, so it's the one worth testing against the broader vocabulary here.

Regions: the SAM3 output (generate_regions_affordance_guided.py) is a flat, overlapping candidate-
mask list (post-NMS but pre-partition), same shape load_regions() normally returns -- so it goes
through partition_regions() here exactly like compute_parts() does for the main pipeline, instead of
being scored as raw overlapping candidates. This script saves its OWN numbered-overlay image (not
the raw-candidate one generate_regions_affordance_guided.py already saved) specifically so the
plotting script's "Part N" always matches what was actually scored, not the pre-partition count --
partition_regions() can merge/drop regions, so the two numberings can legitimately differ.

Usage:
    venv_vlm/bin/python3 label_affordance_guided_synthia.py             # all 5 scenarios
    venv_vlm/bin/python3 label_affordance_guided_synthia.py water_plant  # just this one
"""
import json
import os
import sys

import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoProcessor

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test

from affordance_labeling import (
    VOCABULARIES, build_per_part_question, build_prompt_text, downscale, numbered_overlay,
    partition_regions, score_options,
)
from batch_final_labeling import ALL_MODELS as MODELS, build_per_part_justification_prompt, generate_justification, top1_label
from gpu_contention import wait_for_gpu_headroom
from label_synthia_vocab import build_whole_object_justification_prompt, build_whole_object_question

REGIONS_JSON_DIR = os.path.join(HERE, "json", "affordance_guided_objects_regions")
IMG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "base_images")
OUT_JSON_DIR = os.path.join(HERE, "json", "affordance_guided_objects_justification_qwen3-vl-8b_synthia")
OUT_FIG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "sam_regions_partitioned")

VOCAB = "synthia_filtered"
MODEL_KEY = "qwen3-vl-8b"
SCENARIOS = ["water_plant", "scoop_stir_soup", "support_human_item", "cut_food", "carry_items"]


def compute_parts(scenario_id: str) -> tuple:
    """Mirrors batch_final_labeling.compute_parts(), but reading from this script's own
    region/image locations instead of load_regions()/resolve_image_path()."""
    import numpy as np
    with open(os.path.join(REGIONS_JSON_DIR, f"{scenario_id}.json")) as f:
        regions = json.load(f)
    regions = sorted(regions, key=lambda r: r["area"], reverse=True)
    img_h, img_w = regions[0]["segmentation"]["size"]
    img_path = os.path.join(IMG_DIR, f"{scenario_id}.png")
    image = Image.open(img_path).convert("RGB")
    parts = partition_regions(regions, img_h * img_w, image)
    out = [{
        "region_id": region["region_id"], "area": int(mask.sum()),
        "segmentation": {"size": [img_h, img_w],
                         "counts": mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")},
    } for region, mask in parts]
    print(f"    {len(regions)} SAM3 candidate masks -> {len(out)} disjoint parts")
    return out, img_path


def process_object(model, processor, option_ids, device, scenario_id: str, safe_generate: bool, out_path: str) -> dict:
    parts, img_path = compute_parts(scenario_id)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(img_path).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    overlay.save(os.path.join(OUT_FIG_DIR, f"{scenario_id}_numbered.png"))
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    n_parts = len(parts)

    data = {"object": scenario_id, "model": MODELS[MODEL_KEY]["repo_id"], "image_path": img_path,
            "vocabulary": VOCAB, "regions": [], "overall": None}

    def save():
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    for i, part in enumerate(parts, start=1):
        wait_for_gpu_headroom(log=print)
        fc_prompt = build_prompt_text(processor, build_per_part_question(i, n_parts, options, glosses))
        fc = score_options(model, processor, overlay, fc_prompt, option_ids, device)
        label = top1_label(dict(zip(options, fc.tolist())))

        just_prompt = build_prompt_text(processor, build_per_part_justification_prompt(i, n_parts, label, glosses))
        justification = generate_justification(model, processor, overlay, just_prompt, label, device, safe_generate=safe_generate)

        data["regions"].append({**part, "vocabularies": {VOCAB: {"label": label, "justification": justification}}})
        save()
        print(f"    {part['region_id']:>28s}  {label:>14s}: {justification}")

    wait_for_gpu_headroom(log=print)
    whole_prompt = build_prompt_text(processor, build_whole_object_question(n_parts, options, glosses))
    whole_fc = score_options(model, processor, overlay, whole_prompt, option_ids, device)
    whole_label = top1_label(dict(zip(options, whole_fc.tolist())))
    whole_just_prompt = build_prompt_text(processor, build_whole_object_justification_prompt(n_parts, whole_label, glosses))
    whole_justification = generate_justification(model, processor, overlay, whole_just_prompt, whole_label, device, safe_generate=safe_generate)
    data["overall"] = {"label": whole_label, "justification": whole_justification}
    save()
    print(f"    {'[whole object]':>28s}  {whole_label:>14s}: {whole_justification}")
    return data


def main():
    requested = sys.argv[1:] or SCENARIOS
    os.makedirs(OUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    repo_id, model_cls = MODELS[MODEL_KEY]["repo_id"], MODELS[MODEL_KEY]["model_cls"]
    trust_remote_code = MODELS[MODEL_KEY].get("trust_remote_code", False)
    safe_generate = MODELS[MODEL_KEY].get("safe_generate", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wait_for_gpu_headroom(log=print)
    print(f"Loading {MODEL_KEY} ({repo_id})...")
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    model = model_cls.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code)
    model.eval()
    options = VOCABULARIES[VOCAB]["options"]
    end_token = processor.tokenizer.eos_token
    option_ids = [processor.tokenizer(o + end_token, add_special_tokens=False).input_ids for o in options]
    print(f"vocabulary '{VOCAB}': {len(options)} options")

    for scenario_id in requested:
        wait_for_gpu_headroom(log=print)
        print(f"\n  {scenario_id}:")
        out_path = os.path.join(OUT_JSON_DIR, f"{scenario_id}_affordance_justification.json")
        try:
            process_object(model, processor, option_ids, device, scenario_id, safe_generate, out_path)
        except Exception as e:
            print(f"  {scenario_id}: FAILED -- {e}")
            continue
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
