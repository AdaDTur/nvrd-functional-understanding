"""
Step 3 (the actual test): relabel all 30 novel objects with the curated SYNTHIA-derived vocabulary
("synthia_filtered" in affordance_labeling.py's VOCABULARIES, 230 terms + "none" -- see
filter_synthia_affordances.py / cluster_synthia_affordances.py for how it was built), using
Qwen3-VL-8B-Instruct (qwen3-vl-8b), the same model/scoring method as the rest of this pipeline, so
results are comparable to the existing agd20k-vocabulary runs.

Two question types per object, both saved:
  * per-part (same as batch_final_labeling.py's process_object: numbered_overlay context, "which
    affordance does part N offer")
  * one whole-object question per object -- NEW, not in batch_final_labeling.py. Mirrors the human
    study's overall trial (affordance-study/demo.js's overall_trial / build_overall_images_block):
    same numbered-overlay image, same picker-equivalent vocabulary, but asking about the object's
    functionality as a whole rather than any one numbered part, so model and human answers are
    elicited the same way.

Output goes to its own directory (json/affordance_justification_qwen3-vl-8b_synthia/), separate
from the existing agd20k-vocabulary qwen3-vl-8b output, so this run can't clobber it.

Safety for this run specifically: GPU headroom is re-checked before EVERY PART (not just before
each object, like batch_final_labeling.py does) -- requested explicitly so a contending job
starting on this shared machine mid-object gets caught quickly, not just between objects. Progress
is written to disk after every part too (not just after each object finishes), so killing the
process loses at most one in-flight part's work, never a whole in-progress object.

Usage:
    venv_vlm/bin/python3 label_synthia_vocab.py             # all 30 novel objects
    venv_vlm/bin/python3 label_synthia_vocab.py --resume     # skip objects already fully done
    venv_vlm/bin/python3 label_synthia_vocab.py shape_10 shape_14   # just these
"""
import json
import os
import sys

import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModelForImageTextToText, AutoProcessor

from affordance_labeling import (
    VOCABULARIES, build_per_part_question, build_prompt_text, downscale, numbered_overlay, score_options,
)
from batch_final_labeling import (
    MODELS, build_per_part_justification_prompt, compute_parts, generate_justification, top1_label,
)
from gpu_contention import wait_for_gpu_headroom

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
JSON_DIR = os.path.join(HERE, "json")
OUT_DIR = os.path.join(JSON_DIR, "affordance_justification_qwen3-vl-8b_synthia")
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB = "synthia_filtered"
MODEL_KEY = "qwen3-vl-8b"

OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
           "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
           "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
           "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
           "shape_39", "shape_40"]


def build_whole_object_question(n_parts: int, options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    none_clause = 'If none of these apply, answer "none". ' if "none" in options else ""
    return (
        f"The image shows an object with {n_parts} parts marked and numbered, each outlined and "
        "tinted a different color -- but step back from any individual part. Considering the "
        f"object as a WHOLE, choose the single overall functionality it offers from this list: "
        f"{listed}. {none_clause}Answer with the option only."
    )


def build_whole_object_justification_prompt(n_parts: int, label: str, glosses: dict) -> str:
    gloss = f" ({glosses[label]})" if label in glosses else ""
    return (
        f"The image shows an object with {n_parts} parts marked and numbered, each outlined and "
        f'tinted a different color. Stepping back from individual parts, you determined this '
        f'object\'s overall functionality is "{label}"{gloss}. In one short, specific sentence, '
        "justify this based on the object as a whole (its overall shape, size, or structure). Do "
        "not mention the colors or numbering. Answer with only the justification sentence.\n\n"
        f"{label}:"
    )


def process_object(model, processor, option_ids, device, object_name: str, repo_id: str,
                    safe_generate: bool, out_path: str) -> dict:
    parts, img_path = compute_parts(object_name)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(img_path).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    n_parts = len(parts)

    data = {
        "object": object_name, "model": repo_id, "image_path": img_path, "vocabulary": VOCAB,
        "method": "per_part_with_comparative_context", "regions": [], "overall": None,
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    for i, part in enumerate(parts, start=1):
        wait_for_gpu_headroom(log=print)
        fc_prompt = build_prompt_text(processor, build_per_part_question(i, n_parts, options, glosses))
        fc = score_options(model, processor, overlay, fc_prompt, option_ids, device)
        label = top1_label(dict(zip(options, fc.tolist())))

        just_prompt = build_prompt_text(
            processor, build_per_part_justification_prompt(i, n_parts, label, glosses)
        )
        justification = generate_justification(
            model, processor, overlay, just_prompt, label, device, safe_generate=safe_generate,
        )

        data["regions"].append({**part, "vocabularies": {VOCAB: {"label": label, "justification": justification}}})
        save()
        print(f"    {part['region_id']:>28s}  {label:>14s}: {justification}")

    wait_for_gpu_headroom(log=print)
    whole_prompt = build_prompt_text(processor, build_whole_object_question(n_parts, options, glosses))
    whole_fc = score_options(model, processor, overlay, whole_prompt, option_ids, device)
    whole_label = top1_label(dict(zip(options, whole_fc.tolist())))
    whole_just_prompt = build_prompt_text(
        processor, build_whole_object_justification_prompt(n_parts, whole_label, glosses)
    )
    whole_justification = generate_justification(
        model, processor, overlay, whole_just_prompt, whole_label, device, safe_generate=safe_generate,
    )
    data["overall"] = {"label": whole_label, "justification": whole_justification}
    save()
    print(f"    {'[whole object]':>28s}  {whole_label:>14s}: {whole_justification}")

    return data


def main():
    args = sys.argv[1:]
    resume = "--resume" in args
    args = [a for a in args if a != "--resume"]
    requested = args or OBJECTS

    repo_id, model_cls = MODELS[MODEL_KEY]["repo_id"], MODELS[MODEL_KEY]["model_cls"]
    trust_remote_code = MODELS[MODEL_KEY].get("trust_remote_code", False)
    safe_generate = MODELS[MODEL_KEY].get("safe_generate", False)

    todo = [
        name for name in requested
        if not (resume and os.path.isfile(os.path.join(OUT_DIR, f"{name}_affordance_justification.json")))
    ]
    if not todo:
        print(f"Nothing to do (--resume and every requested object already has an output in {OUT_DIR}).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wait_for_gpu_headroom(log=print)
    print(f"Loading {MODEL_KEY} ({repo_id})...")
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    model = model_cls.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code
    )
    model.eval()
    options = VOCABULARIES[VOCAB]["options"]
    end_token = processor.tokenizer.eos_token
    option_ids = [processor.tokenizer(o + end_token, add_special_tokens=False).input_ids for o in options]
    print(f"vocabulary '{VOCAB}': {len(options)} options")

    for name in todo:
        wait_for_gpu_headroom(log=print)
        print(f"\n  {name}:")
        out_path = os.path.join(OUT_DIR, f"{name}_affordance_justification.json")
        try:
            process_object(model, processor, option_ids, device, name, repo_id, safe_generate, out_path)
        except Exception as e:
            print(f"  {name}: FAILED -- {e}")
            continue
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
