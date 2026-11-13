"""
Standard labeling pipeline: SAM+SLIC regions -> disjoint background-filtered parts -> per-part forced-choice AGD20K
label + one-sentence justification.

Each part is an independent model call (no autoregressive copy-paste between parts -- the failure mode that made an
all-parts-in-one-generation version collapse to a single repeated label), but every call sees the same numbered,
coloured overlay of all the object's parts, so the model can contrast part i with the others (the fix for
object-level language-prior collapse, e.g. corkscrew "open" x19). Confirmed on the objects that had collapsed.

Parts are recomputed from the region files (load_regions -> partition_regions), capped to the MAX_REGIONS largest
regions since some shape-texture objects segment into 100+ same-material slivers. Output goes to
json/affordance_justification/<object>_affordance_justification.json; visualize_affordance_justification.py and
visualize_affordance_labels_only.py read it, as do rise/loo_affordance_verbs.py (which reuse its parts).

Usage:
    python batch_final_labeling.py [object ...]                   # default model: Qwen2-VL-7B (all 33 objects)
    python batch_final_labeling.py --model qwen3-vl-8b [object ...]   # or internvl3-8b -- see MODELS below
    python batch_final_labeling.py --model qwen3-vl-8b --resume    # skip objects that already have an output
Every model in MODELS runs the identical forced-choice-over-options prompt/scoring; the default
model's outputs stay at json/affordance_justification/ (what every downstream script reads), other
models write to their own json/affordance_justification_<model_key>/ sibling directory.
Without --resume (the default), existing outputs are overwritten, same as before --resume existed.

Two scoring modes per part, both keeping the model's top-10 (TOP_K) answers, alongside the existing
"label"/"justification" (unchanged: label is still the log-prob top-1, so every downstream reader
works as before):
  vocabularies[<vocab>]["logprob"]["ranked"]     -- the 10 highest-scoring options from score_options,
      each {"option", "logprob", "prob"} (prob = softmax over ALL options). Open-weight models only.
  vocabularies[<vocab>]["generation"]            -- the model asked to WRITE its 10 most plausible
      options ranked (build_ranked_question), greedy-decoded and mapped onto the vocabulary:
      {"ranked": [...], "n_valid", "invalid": [unmatched lines], "raw": <the model's text>}. Same
      prompt as label_gpt_api.py's, so this mode is comparable across open and closed models.
With --resume, any object whose output already has both top-10 modes for every part is skipped --
for long/full-dataset runs under labeling_watchdog.sh (which restarts on an unexpected kill and
would otherwise redo completed objects from scratch every time).
"""
import os
import sys
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
try:  # gone from transformers 5.x; only the legacy qwen2-vl-7b entry needs it
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = AutoModelForImageTextToText

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from affordance_labeling import (
    AFFORDANCE_JUSTIFICATION_DIR, JSON_DIR, MODEL_NAME, TOP_K, VOCABULARIES, build_per_part_question, build_prompt_text,
    build_ranked_question, downscale, first_sentence, load_regions, numbered_overlay, parse_ranked, partition_regions,
    processor_inputs, resolve_image_path, score_options, topk_from_scores,
)
from gpu_contention import contention_blockers, wait_for_gpu_headroom

HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
VOCAB = "agd20k"
MAX_NEW_TOKENS = 64
RANKED_MAX_NEW_TOKENS = 200
YIELD_EXIT_CODE = 75  # "yielded the GPU to another user" -- labeling_watchdog.sh restarts without counting a retry  # TOP_K numbered options, some multi-word (synthia_filtered)
MAX_REGIONS = 30
REGION_SOURCE = "sam_slic"

# Additional open-source VLMs run through the IDENTICAL forced-choice-over-options pipeline as the
# default Qwen2-VL-7B baseline -- prompted zero-shot on this affordance vocabulary, not fine-tuned on
# it, same as the default -- so GAT (the trained specialist, run_gat_inference.py) can be compared
# against more than one untrained VLM instead of just one. GLM-5.2's vision line ("GLM-5V") turned out
# not to have an open release at all, so it's not in this roster; Kimi-VL-A3B was tried and dropped
# after 5 separate transformers-version compatibility shims still didn't clear it (each fix exposed a
# new legacy Cache-API usage in its bundled trust_remote_code modeling file) -- not worth chasing
# further with 4 other models already clean.
# trust_remote_code: a config.json listing a plausible-looking "...ForConditionalGeneration"
# architecture name does NOT mean that class is actually registered in the installed transformers --
# molmo2-8b looked natively supported by that signal and wasn't; transformers itself raised "contains
# custom code which must be executed". It's from Allen AI's official org, so running its bundled code
# is an acceptable risk here -- but don't assume the next addition is native without actually trying
# to load it first.
# Study roster (cut down 2026-11-05): one ~4B checkpoint per family, so the open models are
# size-matched. Phi-3.5's only vision model is 4.2B, which pins the tier; every other family is picked
# at its nearest size (Gemma-4 E4B: 4.5B effective / 8B incl. per-layer embeddings). GPT-5.4 is
# closed (no published size, no logprobs) and runs separately via label_gpt_api.py.
#
# Per-model keys beyond repo_id/model_cls/trust_remote_code/safe_generate:
#   venv: which venv can load it. Gemma-4 and Qwen3.5 need transformers>=5.5 (venv_vlm5); Molmo2's
#     bundled remote code targets transformers 4.57 (venv_vlm); Phi-3.5's targets the pre-4.47 cache
#     API (calls DynamicCache.get_usable_length), so it gets venv_phi (transformers 4.46.3).
#   end_token: the token that closes an assistant turn, appended to every option before scoring. Not
#     always eos_token -- Gemma-4's eos is <eos> but turns end with <turn|>; Phi-3.5's eos is
#     <|endoftext|> but turns end with <|end|>. Defaults to eos_token when unset.
#   copy_cache / continuation_mask: see score_options -- both needed for Qwen3.5 (recurrent
#     linear-attention state can't be cropped; M-RoPE mis-sizes positions given a full-length mask).
#   prompt_style / chat_template_kwargs: see build_prompt_text.
#   attn_implementation: Phi-3.5's config defaults to flash_attention_2, which isn't installed.
MODELS = {
    # safe_generate: repetition_penalty/no_repeat_ngram_size's logits processors gather over the FULL
    # input_ids history (prompt + image placeholder tokens included) as indices into the vocab-sized
    # logits tensor. Molmo2's image placeholder token ids fall outside its text vocab range, so that
    # gather reads out of bounds and crashes CUDA with a device-side assert -- confirmed via a direct
    # repro isolating the failure to this exact call. Dropping those two knobs (plain greedy decoding
    # instead) avoids it. Phi-3.5 gets the same treatment: its processor marks image positions with
    # negative input_ids, which hit the same gather.
    "molmo2-4b": {
        "repo_id": "allenai/Molmo2-4B", "model_cls": AutoModelForImageTextToText, "trust_remote_code": True,
        "safe_generate": True, "venv": "venv_vlm",
    },
    "phi3.5-vision": {
        "repo_id": "microsoft/Phi-3.5-vision-instruct", "model_cls": AutoModelForCausalLM,
        "trust_remote_code": True, "safe_generate": True, "venv": "venv_phi", "end_token": "<|end|>",
        "prompt_style": "phi3v", "attn_implementation": "eager", "processor_kwargs": {"num_crops": 4},
    },
    "gemma4-e4b": {
        "repo_id": "google/gemma-4-E4B-it", "model_cls": AutoModelForImageTextToText, "venv": "venv_vlm5",
        "end_token": "<turn|>",
        "copy_cache": True,  # its sliding-window cache layers refuse crop() back to a prefix length
    },
    "qwen3.5-4b": {
        "repo_id": "Qwen/Qwen3.5-4B", "model_cls": AutoModelForImageTextToText, "venv": "venv_vlm5",
        "copy_cache": True, "continuation_mask": False, "chat_template_kwargs": {"enable_thinking": False},
    },
}

# Previous roster -- no longer part of the study, kept so existing outputs stay reproducible and
# older scripts (label_synthia_vocab.py, relabel_none_cases.py, ...) that look these keys up keep working.
LEGACY_MODELS = {
    "qwen2-vl-7b": {"repo_id": MODEL_NAME, "model_cls": Qwen2VLForConditionalGeneration},
    "qwen3-vl-8b": {"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "model_cls": AutoModelForImageTextToText},
    "internvl3-8b": {"repo_id": "OpenGVLab/InternVL3-8B-hf", "model_cls": AutoModelForImageTextToText},
    "molmo2-8b": {
        "repo_id": "allenai/Molmo2-8B", "model_cls": AutoModelForImageTextToText, "trust_remote_code": True,
        "safe_generate": True,
    },
}
ALL_MODELS = {**LEGACY_MODELS, **MODELS}

OBJECTS = [
    ("known", ["ladle", "chair", "corkscrew", "backpack", "bow-tie", "clock", "coffee-table", "fedora",
               "fire-alarm", "wallet"]),
    ("novel", ["shape_16", "shape_27", "shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12",
               "shape_13", "shape_14", "shape_15", "shape_17", "shape_18", "shape_19", "shape_20",
               "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_28",
               "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38", "shape_39",
               "shape_40"]),  # all 30 novel objects (ids scattered 1-40, see list_novel_objects.py
               # in regions_test/ for how the full set was determined)
    ("shape-shape", ["backpack_amplifier", "bonsai_amplifier", "bonsai_crystal", "cactus_camera",
                      "compass_totem", "crystal_telescope", "dagger_amplifier", "dagger_projector",
                      "drone_dagger"]),
    ("shape-texture", ["bird-feathers_headphones", "bird-feathers_saxophone", "checkerboard_tv-remote",
                        "crochet_broom"]),
]


def build_per_part_justification_prompt(part_idx: int, n_parts: int, label: str, glosses: dict) -> str:
    gloss = f" ({glosses[label]})" if label in glosses else ""
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        f"outlined and tinted a different color. You determined that part {part_idx}'s affordance is "
        f'"{label}"{gloss}. In one short, specific sentence, justify this based on what you observe '
        f"about part {part_idx} specifically (its shape, material, or position) -- and why it might "
        "differ from the other numbered parts. Do not mention the colors or numbering. Answer with "
        f"only the justification sentence.\n\n{label}:"
    )


@torch.no_grad()
def generate_justification(
    model, processor, image: Image.Image, prompt_text: str, label: str, device, safe_generate: bool = False,
) -> str:
    forced_prefix = f"{label}:"
    inputs = processor_inputs(processor, prompt_text + forced_prefix, image, device)
    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    if not safe_generate:  # see MODELS["molmo2-4b"]'s comment for why this is skipped for that model
        gen_kwargs.update(repetition_penalty=1.3, no_repeat_ngram_size=3)
    out = model.generate(**inputs, **gen_kwargs)
    continuation = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return first_sentence(continuation)


@torch.no_grad()
def generate_ranked(model, processor, image: Image.Image, prompt_text: str, options: list, device,
                    stop_ids: "list | None" = None) -> dict:
    """Direct-generation mode: greedy-decode the model's own ranked top-K list (build_ranked_question)
    and map it onto the vocabulary (parse_ranked). Plain greedy for every model -- the
    repetition_penalty / no_repeat_ngram_size that generate_justification uses would penalize the
    list's own numbering ("1.", "2.", ... share tokens) and the vocabulary words the prompt itself
    lists, distorting exactly what this records.

    stop_ids: token ids that end generation, in addition to the model's default eos. Needed when a
    model's turn ends with a token other than eos_token (Phi-3.5: <|end|> vs <|endoftext|>) -- without
    it Phi ran straight past its answer into invented follow-up conversations (212/215 parts)."""
    inputs = processor_inputs(processor, prompt_text, image, device)
    gen_kw = {"eos_token_id": stop_ids} if stop_ids else {}
    out = model.generate(**inputs, max_new_tokens=RANKED_MAX_NEW_TOKENS, do_sample=False, **gen_kw)
    raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {**parse_ranked(raw, options), "raw": raw.strip()}


def has_topk_fields(path: str, vocab: str, modes: tuple = ("logprob", "generation")) -> bool:
    """--resume's "done" test: an output counts as done only if every part already carries both
    top-K modes -- outputs from before top-K collection existed get redone, not skipped."""
    try:
        with open(path) as f:
            regions = json.load(f)["regions"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(regions) and all(
        set(modes) <= set(r.get("vocabularies", {}).get(vocab, {})) for r in regions
    )


def top1_label(fc_logprobs: dict) -> str:
    return max(fc_logprobs, key=fc_logprobs.get)


def compute_parts(object_name: str) -> tuple:
    """Disjoint parts as (region_id, area, RLE segmentation) dicts, plus the image path."""
    regions = load_regions(object_name, MAX_REGIONS, REGION_SOURCE)
    img_h, img_w = regions[0]["segmentation"]["size"]
    img_path = resolve_image_path(object_name)
    parts = partition_regions(regions, img_h * img_w, Image.open(img_path).convert("RGB"))
    out = [{
        "region_id": region["region_id"], "area": int(mask.sum()),
        "segmentation": {"size": [img_h, img_w],
                         "counts": mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")},
    } for region, mask in parts]
    print(f"    {len(regions)} regions (capped) -> {len(out)} disjoint parts")
    return out, img_path


def process_object(
    model, processor, option_ids, device, object_name: str, repo_id: str, safe_generate: bool = False,
    cfg: "dict | None" = None, get_model=None,
) -> dict:
    """get_model: optional callable run before every part that returns the model to use -- main()
    passes one that unloads the model under GPU contention and reloads it once clear. When given,
    `model` is ignored and no reference to a model is held across parts (a reference kept here
    would stop the unload from actually freeing GPU memory)."""
    cfg = cfg or {}
    prompt_kw = dict(prompt_style=cfg.get("prompt_style", "default"),
                     chat_template_kwargs=cfg.get("chat_template_kwargs"))
    parts, img_path = compute_parts(object_name)
    masks = [mask_utils.decode(p["segmentation"]).astype(bool) for p in parts]
    image = Image.open(img_path).convert("RGB")
    overlay = downscale(numbered_overlay(image, masks))
    options = VOCABULARIES[VOCAB]["options"]
    glosses = VOCABULARIES[VOCAB]["glosses"]
    tok = processor.tokenizer
    stop_ids = sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids(cfg.get("end_token") or tok.eos_token))
                       if isinstance(i, int) and i != tok.unk_token_id})

    results = []
    if get_model is not None:
        model = None
    for i, part in enumerate(parts, start=1):
        if get_model is not None:
            model = get_model()
        fc_prompt = build_prompt_text(processor, build_per_part_question(i, len(parts), options, glosses), **prompt_kw)
        fc = score_options(model, processor, overlay, fc_prompt, option_ids, device,
                           copy_cache=cfg.get("copy_cache", False),
                           continuation_mask=cfg.get("continuation_mask", True))
        label = top1_label(dict(zip(options, fc.tolist())))

        just_prompt = build_prompt_text(
            processor, build_per_part_justification_prompt(i, len(parts), label, glosses), **prompt_kw
        )
        justification = generate_justification(
            model, processor, overlay, just_prompt, label, device, safe_generate=safe_generate,
        )

        rank_prompt = build_prompt_text(processor, build_ranked_question(i, len(parts), options, glosses), **prompt_kw)
        generation = generate_ranked(model, processor, overlay, rank_prompt, options, device, stop_ids=stop_ids)

        results.append({**part, "vocabularies": {VOCAB: {
            "label": label, "justification": justification,
            "logprob": {"ranked": topk_from_scores(options, fc)},
            "generation": generation,
        }}})
        gen_top = generation["ranked"][0] if generation["ranked"] else "-"
        print(f"    {part['region_id']:>28s}  {label:>14s}: {justification}")
        print(f"    {'':>28s}  generation top-1 {gen_top!r}, {generation['n_valid']}/{TOP_K} valid")
        if get_model is not None:
            model = None  # drop our reference so get_model() can really free it next part

    return {
        "object": object_name, "model": repo_id, "image_path": img_path,
        "region_source": REGION_SOURCE, "method": "per_part_with_comparative_context",
        "top_k": TOP_K, "modes": ["logprob", "generation"],
        "regions": results,
    }


def main():
    args = sys.argv[1:]
    resume = "--resume" in args  # skip objects whose output file already exists, instead of overwriting
    args = [a for a in args if a != "--resume"]
    model_key = "qwen2-vl-7b"
    if args and args[0] == "--model":
        model_key = args[1]
        args = args[2:]
    split = None  # --split novel: every object in that OBJECTS category (instead of naming them)
    if args and args[0] == "--split":
        split = args[1]
        args = args[2:]
    if model_key not in ALL_MODELS:
        print(f"Unknown --model {model_key!r}; study roster: {sorted(MODELS)}; legacy: {sorted(LEGACY_MODELS)}")
        return
    cfg = ALL_MODELS[model_key]
    repo_id, model_cls = cfg["repo_id"], cfg["model_cls"]
    trust_remote_code = cfg.get("trust_remote_code", False)
    safe_generate = cfg.get("safe_generate", False)

    # The default model's outputs stay at the original path (everything downstream -- rise/loo scripts,
    # visualize_affordance_*.py -- reads from there); every other model gets its own sibling directory
    # so multi-model runs don't clobber each other or the default.
    out_dir = AFFORDANCE_JUSTIFICATION_DIR if model_key == "qwen2-vl-7b" else os.path.join(
        JSON_DIR, f"affordance_justification_{model_key}"
    )
    os.makedirs(out_dir, exist_ok=True)

    requested = args
    todo = [
        (category, name) for category, names in OBJECTS for name in names
        if (not requested or name in requested) and (split is None or category == split)
        and not (resume and has_topk_fields(os.path.join(out_dir, f"{name}_affordance_justification.json"), VOCAB))
    ]
    if not todo:
        print(f"Nothing to do (--resume and every requested object already has an output in {out_dir}).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_key} ({repo_id})...")
    processor = AutoProcessor.from_pretrained(
        repo_id, trust_remote_code=trust_remote_code, **cfg.get("processor_kwargs", {})
    )
    def load_model():
        load_kw = {}
        if "attn_implementation" in cfg:
            # Phi-3.5's config.json hardcodes _attn_implementation=flash_attention_2 and its remote
            # __init__ reads it back off the config, so the from_pretrained kwarg alone doesn't stick.
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
            model_config._attn_implementation = cfg["attn_implementation"]
            load_kw = {"config": model_config, "attn_implementation": cfg["attn_implementation"]}
        m = model_cls.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust_remote_code, **load_kw
        )
        return m.eval()

    # Yield the GPU to other users: before every part, if contention_blockers() reports anything on
    # our visible GPUs, EXIT with YIELD_EXIT_CODE instead of waiting while holding the card.
    # Unloading in-process doesn't work: transformers' load reserves the weights as one large
    # allocator block, and a few MB of leftover tensors pin it, so empty_cache() returned nothing
    # (tested: 9.2 GB stayed reserved after the model was deleted). Exiting frees everything;
    # labeling_watchdog.sh treats this exit code as a pause, not a crash, and restarts with
    # --resume, where the initial wait_for_gpu_headroom() below holds off until the GPU is clear.
    # At most the in-progress object is redone.
    state = {"model": None}

    def get_model():
        if state["model"] is not None:
            blockers = contention_blockers()
            if blockers:
                print(f"  GPU contention {blockers} -- exiting to release the GPU "
                      f"(watchdog resumes once clear)", flush=True)
                sys.exit(YIELD_EXIT_CODE)
            return state["model"]
        wait_for_gpu_headroom(log=print)
        print(f"  loading {model_key}...", flush=True)
        state["model"] = load_model()
        return state["model"]

    get_model()  # initial load, after waiting for headroom
    options = VOCABULARIES[VOCAB]["options"]
    # Not hardcoded to Qwen's "<|im_end|>" -- read off each model's own tokenizer so this generalizes
    # to the other models in MODELS (verified all three currently use "<|im_end|>" as eos_token, but
    # this doesn't assume that holds for whatever gets added next).
    end_token = cfg.get("end_token") or processor.tokenizer.eos_token
    end_ids = processor.tokenizer(end_token, add_special_tokens=False).input_ids
    assert len(end_ids) == 1, f"end_token {end_token!r} isn't a single token for {repo_id}: {end_ids}"
    option_ids = [processor.tokenizer(o + end_token, add_special_tokens=False).input_ids for o in options]

    last_category = None
    for category, name in todo:
        if category != last_category:
            print(f"\n=== {category} ===")
            last_category = category
        # Contention is checked (and the GPU released if needed) before every PART via get_model,
        # not just once per object, so another user's job gets the card back within ~a part's time.
        print(f"  {name}:")
        try:
            data = process_object(None, processor, option_ids, device, name, repo_id, safe_generate=safe_generate,
                                  cfg=cfg, get_model=get_model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {name}: FAILED -- {e}")
            continue
        out_path = os.path.join(out_dir, f"{name}_affordance_justification.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_distinct = len({r["vocabularies"][VOCAB]["label"] for r in data["regions"]})
        print(f"  saved {out_path}  (distinct labels: {n_distinct} of {len(data['regions'])} parts)")


if __name__ == "__main__":
    main()
