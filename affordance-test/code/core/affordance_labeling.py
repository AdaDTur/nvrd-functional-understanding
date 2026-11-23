"""
Shared core for the training-free per-part affordance labelling pipeline (Qwen2-VL, AGD20K verbs).

  * Vocabulary + prompts:  AGD20K_VERBS (+ "wear", + "none"), build_question, build_per_part_question,
    build_prompt_text
  * Parts:  partition_regions -- SAM regions are often nested (a whole-object region containing its parts), so a
    "handle" never exists as its own region. Each pixel is assigned to the smallest foreground region containing
    it, giving disjoint parts; background-sized and background-coloured regions (negative space) are dropped.
  * Images:  downscale (MAX_SIDE), highlight_region (single-part highlight), numbered_overlay (all parts coloured,
    outlined and numbered)
  * Scoring:  score_options -- teacher-forced log P(option + <|im_end|> | image, prompt) for every option via one
    shared KV-cache prefix, so we get a full distribution instead of one sampled guess. Verified against a
    cache-free full-forward reference in verify_methods.py. topk_from_scores keeps its TOP_K best options.
  * Direct generation:  build_ranked_question asks the model to write out its TOP_K ranked options;
    parse_ranked / match_ranked map that answer back onto the vocabulary. Works for any model,
    including API models with no logprobs.

Images are downscaled to MAX_SIDE px: the shared GPUs are often too full for full-resolution vision-tower
activations, and answers are somewhat resolution-sensitive, so compare runs at equal MAX_SIDE.

Entry points live in other files (batch_final_labeling.py etc.); this module has no CLI and parses no argv.
"""

import copy
import inspect
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import DynamicCache


_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
REPO_ROOT = os.path.dirname(HERE)
REGIONS_TEST = os.path.join(REPO_ROOT, "regions_test")
# Output JSONs are organized by which pipeline stage produced them; imported by every script that reads or writes
# one of these, so the layout only needs to change in one place.
JSON_DIR = os.path.join(HERE, "json")
AFFORDANCE_JUSTIFICATION_DIR = os.path.join(JSON_DIR, "affordance_justification")  # final, standard results
os.makedirs(AFFORDANCE_JUSTIFICATION_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # legacy; kept only for the "qwen2-vl-7b" MODELS entry, not run for the study
MASK_SUBDIRS = {"sam": "sam_masks", "sam_slic": "sam_slic_masks"}
BACKGROUND_AREA_FRAC = 0.5
MIN_PART_FRAC = 0.001
DIM_FACTOR = 0.3
CONTOUR_RGB = (255, 0, 255)
END_TOKEN = "<|im_end|>"
MAX_SIDE = 512
TAB10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
         (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


AGD20K_VERBS = [
    "beat", "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", "drink_with", "eat",
    "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel", "pick_up",
    "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo", "talk_on",
    "text_on", "throw", "type_on", "wash", "write",
    # Additions beyond the standard AGD20K/OOAL 36-verb set, made for this project -- see
    # VOCABULARY_REVIEW.md for the running list of gaps found and reasoning behind each addition.
    "wear",
]

def _load_synthia_filtered_verbs() -> list:
    # Built by filter_synthia_affordances.py (relevance filter, 661 -> 473) then
    # cluster_synthia_affordances.py (synonym merge, 473 -> 252), then hand-edited down to 230 --
    # see synthia_affordances_clustered.txt's git history / the hand edit for what was cut.
    path = os.path.join(HERE, "synthia_affordances_clustered.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


VOCABULARIES = {
    "agd20k": {"options": [v.replace("_", " ") for v in AGD20K_VERBS] + ["none"], "glosses": {}},
    # Same options, minus "none" -- forces the model to commit to one of the real verbs instead of
    # abstaining. Used for targeted re-scoring of parts that came back "none" under the standard
    # vocabulary above (see relabel_none_cases.py), not for a full pipeline run.
    "agd20k_no_none": {"options": [v.replace("_", " ") for v in AGD20K_VERBS], "glosses": {}},
    # Curated replacement/extension for agd20k's 36 verbs, meant to fix AGD20K's ambiguous/missing
    # coverage -- see filter_synthia_affordances.py and cluster_synthia_affordances.py for how this
    # was derived from the 661-term SYNTHIA ontology. "none" is included for parity with "agd20k"
    # (same fallback semantics), even though the much broader coverage here should need it less.
    "synthia_filtered": {"options": _load_synthia_filtered_verbs() + ["none"], "glosses": {}},
}


HIGHLIGHT_INTRO = (
    "The image shows an object. One part of it is highlighted: it is at full brightness and "
    "outlined in magenta, while the rest of the image is darkened (use it only as context). "
)


def build_question(options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    return (
        HIGHLIGHT_INTRO
        + "Which single affordance does the highlighted part offer? "
        + f"Choose exactly one of: {listed}. "
        + 'If the highlighted part offers none of these (for example plain background), answer "none". '
        + "Answer with the option only."
    )


def build_prompt_text(processor, question: str, prompt_style: str = "default",
                      chat_template_kwargs: "dict | None" = None) -> str:
    """prompt_style "phi3v": Phi-3.5-vision's chat template only takes string content and expects a
    literal <|image_1|> placeholder, not a structured {"type": "image"} entry. chat_template_kwargs is
    forwarded to apply_chat_template -- e.g. {"enable_thinking": False} for Qwen3.5, whose template
    otherwise opens a <think> block that the forced-choice scoring would land inside of."""
    if prompt_style == "phi3v":
        messages = [{"role": "user", "content": "<|image_1|>\n" + question}]
        templater = processor.tokenizer  # Phi3VProcessor carries no chat_template of its own
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        templater = processor
    return templater.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(chat_template_kwargs or {})
    )


def downscale(img: Image.Image) -> Image.Image:
    scale = MAX_SIDE / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def resolve_image_path(object_name: str) -> str:
    """Looks for object_name's image first among NVRD splits (regions_test/data/<split>/),
    then in test_images/, trying common extensions."""
    search_dirs = [
        os.path.join(REGIONS_TEST, "data", split)
        for split in ("novel", "known", "shape-shape", "shape-texture")
    ] + [os.path.join(REPO_ROOT, "test_images")]
    for d in search_dirs:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(d, f"{object_name}{ext}")
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not find an image for '{object_name}' under regions_test/data/*/ or test_images/"
    )


def load_regions(object_name: str, max_regions: "int | None" = None, source: str = "sam"):
    """source: 'sam' (SAM masks only) or 'sam_slic' (SAM masks + SLIC superpixels filling the pixels
    SAM left unmasked, from regions/region_features/slic/merge_sam_and_slic_regions.py)."""
    sam_path = os.path.join(REGIONS_TEST, "output", MASK_SUBDIRS[source], f"{object_name}.json")
    with open(sam_path) as f:
        regions = json.load(f)
    regions = sorted(regions, key=lambda r: r["area"], reverse=True)
    if max_regions is not None:
        regions = regions[:max_regions]
    return regions


def highlight_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(image).astype(np.float32)
    arr[~mask] *= DIM_FACTOR
    out = np.ascontiguousarray(arr.astype(np.uint8))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, round(min(mask.shape) / 300))
    cv2.drawContours(out, contours, -1, CONTOUR_RGB, thickness)
    return Image.fromarray(out)

BG_COLOR_TOLERANCE = 20  # max per-pixel RGB distance from the image's own background to still count
                         # as "background-colored"
BG_COLOR_FRAC_THRESHOLD = 0.85  # a part is dropped as negative space if at least this fraction of its
                                # pixels are background-colored
BORDER_BG_COLOR_FRAC_THRESHOLD = 0.45  # looser threshold used ONLY for regions touching the image
    # edge (see below) -- the studio floor/backdrop often picks up a soft reflection of the object
    # (confirmed directly: coffee-table's floor region was 73% exact-background-color but the other
    # ~27% was genuine brown table-wood RGB from its reflection, spanning almost the full width under
    # the table), which the strict 0.85 threshold doesn't tolerate. Verified safe on 6 objects
    # (coffee-table, clock, fire-alarm, shape_11/13/21/22): every large region that was actually
    # background/floor had a bg-color fraction of 0.50-0.73; the one confirmed false-positive risk
    # (clock's white face, 0.61) does NOT touch the image border, so gating the looser threshold on
    # border-touching keeps that legitimate part while catching all the confirmed background cases.


def estimate_background_color(image: Image.Image) -> np.ndarray:
    """These are studio product photos on a plain, near-uniform backdrop -- a thin border strip is
    almost always pure background (the object is centered and never spans the full frame), so its
    median color is a robust background estimate without needing any segmentation."""
    arr = np.array(image)
    h, w = arr.shape[:2]
    m = max(2, round(min(h, w) * 0.02))
    border = np.concatenate([arr[:m].reshape(-1, 3), arr[-m:].reshape(-1, 3),
                              arr[:, :m].reshape(-1, 3), arr[:, -m:].reshape(-1, 3)])
    return np.median(border, axis=0)


def is_background_like(arr: np.ndarray, mask: np.ndarray, bg_color: np.ndarray) -> bool:
    """True if a part is mostly background color showing through a hole/gap in the object (e.g. the
    open space in a chair's back, or a corkscrew's finger-holes), or is the backdrop/floor itself
    (possibly with some object reflection bled in), rather than actual object material -- SAM2
    segments these the same as any other region since they're just as geometrically coherent, but
    they aren't parts of the object at all and should never be scored for affordance.

    Floor/backdrop regions always touch the image's outer edge (the object is centered and never
    spans the full frame); a real object part never does, even a pale one like a white clock face.
    So a region touching the border gets a looser threshold, tolerant of reflection/shadow noise;
    one that doesn't touch the border keeps the strict threshold, so a legitimately pale interior
    part isn't mistaken for background just because its color happens to resemble the backdrop."""
    touches_border = mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    threshold = BORDER_BG_COLOR_FRAC_THRESHOLD if touches_border else BG_COLOR_FRAC_THRESHOLD
    dist = np.linalg.norm(arr[mask].astype(np.float32) - bg_color, axis=-1)
    return float((dist < BG_COLOR_TOLERANCE).mean()) >= threshold


def partition_regions(regions: list, img_area: int, image: Image.Image) -> list:
    """Nested/overlapping SAM masks -> disjoint parts: every pixel goes to the smallest foreground
    region containing it. Residuals are cleaned by a morphological opening (removes the thin slivers
    left where nested masks don't line up exactly), components under MIN_PART_FRAC of the image are
    dropped, and components that are mostly background color (negative space, not material) are
    dropped regardless of size. Returns [(source_region, part_mask)], largest source region first."""
    foreground = sorted(
        (r for r in regions if r["area"] < BACKGROUND_AREA_FRAC * img_area), key=lambda r: r["area"], reverse=True
    )
    decoded = [mask_utils.decode(r["segmentation"]).astype(bool) for r in foreground]
    label_map = np.full(decoded[0].shape, -1, dtype=np.int32)
    for i, m in enumerate(decoded):  # largest first, so smaller regions overwrite their parents
        label_map[m] = i
    k = max(3, round(min(label_map.shape) * 0.006)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    arr = np.array(image)
    bg_color = estimate_background_color(image)
    parts = []
    for i, region in enumerate(foreground):
        opened = cv2.morphologyEx((label_map == i).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, comp = cv2.connectedComponents(opened)
        keep = np.zeros(opened.shape, dtype=bool)
        for c in range(1, n):
            component = comp == c
            if component.sum() >= MIN_PART_FRAC * img_area and not is_background_like(arr, component, bg_color):
                keep |= component
        if keep.any():
            parts.append((region, keep))
    return parts


def processor_inputs(processor, prompt_text: str, image: Image.Image, device) -> dict:
    """processor(text, image) -> device tensors. Phi-3.5-vision's processor rejects a list for text
    (it regex-scans a single string for <|image_N|> tags), so it gets the bare string."""
    text = prompt_text if type(processor).__name__.startswith("Phi3V") else [prompt_text]
    inputs = processor(text=text, images=[image], return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _supports_cache_position(model) -> bool:
    """Whether model.forward() accepts cache_position -- most do, but a custom (trust_remote_code)
    forward() isn't guaranteed to: kimi-vl-a3b's forward() raised a hard TypeError on it ("unexpected
    keyword argument"), confirming its signature has neither that named parameter nor a **kwargs
    catch-all. Checked once per model rather than hardcoding a list of which models lack it, so this
    generalizes to whatever gets added to MODELS next without another signature-mismatch surprise."""
    params = inspect.signature(model.forward).parameters
    return "cache_position" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


@torch.no_grad()
def score_options(model, processor, image: Image.Image, prompt_text: str, option_ids: list, device,
                  copy_cache: bool = False, continuation_mask: bool = True) -> np.ndarray:
    """log P(option + <end-of-turn> | prompt, image) for every option. The image+prompt prefix is run
    once; each option's continuation reuses that KV cache, which is cropped back afterwards.

    copy_cache: instead of cropping, run each option on a deepcopy of the prefix cache. Needed for
    hybrid models whose cache holds recurrent state (Qwen3.5's linear-attention layers) -- that state
    has no notion of "the first prompt_len tokens", so crop() can't rewind it and every option after
    the first would silently be scored on top of the previous option's tokens.

    continuation_mask=False: omit attention_mask on the cached continuation forward. Qwen3.5's M-RoPE
    derives continuation position ids from attention_mask.cumsum() when a mask is given, which yields
    prompt_len+n positions for an n-token continuation (shape error); without a mask it uses
    arange(past_len, past_len+n) + its stored rope_deltas, the correct positions. Safe here because
    the batch is a single unpadded sequence."""
    inputs = processor_inputs(processor, prompt_text, image, device)
    prompt_len = inputs["input_ids"].shape[1]
    # copy_cache models get whatever cache class the model builds for itself (hybrid architectures
    # need their own, not a plain DynamicCache); the default path keeps the original behavior.
    cache = None if copy_cache else DynamicCache()
    out = model(**inputs, past_key_values=cache, use_cache=True)
    if copy_cache:
        cache = out.past_key_values
    first_logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    use_cache_position = _supports_cache_position(model)

    scores = []
    for ids in option_ids:
        lp = first_logp[ids[0]].item()
        if len(ids) > 1:
            cont = torch.tensor([ids[:-1]], device=device)
            n = cont.shape[1]
            opt_cache = copy.deepcopy(cache) if copy_cache else cache
            kwargs = dict(input_ids=cont, past_key_values=opt_cache, use_cache=True)
            if continuation_mask:
                kwargs["attention_mask"] = torch.ones(1, prompt_len + n, dtype=torch.long, device=device)
            if use_cache_position:
                kwargs["cache_position"] = torch.arange(prompt_len, prompt_len + n, device=device)
            o = model(**kwargs)
            logp = F.log_softmax(o.logits[0].float(), dim=-1)
            lp += sum(logp[j, ids[j + 1]].item() for j in range(n))
            if not copy_cache:
                cache.crop(prompt_len)
        scores.append(lp)
    return np.array(scores)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


TOP_K = 10  # how many ranked responses both scoring modes record per part


def topk_from_scores(options: list, scores: np.ndarray, k: int = TOP_K) -> list:
    """Log-prob mode's top-k: the k highest-scoring options from score_options(), best first.
    "logprob" is the raw sequence log P(option + end token); "prob" is that renormalized over the
    full option list (softmax), so it sums to 1 across ALL options, not just the k kept here."""
    probs = softmax(scores)
    order = np.argsort(-scores)[:k]
    return [{"option": options[j], "logprob": float(scores[j]), "prob": float(probs[j])} for j in order]


def build_ranked_question(part_idx: int, n_parts: int, options: list, glosses: dict, k: int = TOP_K,
                          answer_format: str = "lines") -> str:
    """Direct-generation mode's question: same framing as build_per_part_question, but asks for the
    k most plausible options ranked, instead of one. answer_format "lines" (open models: a numbered
    plain-text list, parsed by parse_ranked) or "json" (API models whose structured output schema
    already constrains the shape -- see label_gpt_api.py)."""
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    # "rank none first" alone let models stop at a bare "none" (qwen3.5-4b did on 2 of 5 shape_16
    # parts), so the clause spells out that the rest of the list is still required.
    none_clause = (f'If part {part_idx} offers none of these, rank "none" first, then still list the '
                   f"{k - 1} next most plausible options. " if "none" in options else "")
    if answer_format == "lines":
        fmt = (f"Answer with exactly {k} different options from the list, one per line, numbered 1 to "
               f"{k}, most likely first, and nothing else.")
    else:
        fmt = f"Give exactly {k} different options from the list, most likely first."
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        f"outlined and tinted a different color. Focusing specifically on part {part_idx}: consider "
        "whether its affordance differs from the other numbered parts, then rank the "
        f"{k} affordances part {part_idx} most plausibly offers, from this list: {listed}. "
        f"{none_clause}{fmt}"
    )


def _normalize_option(text: str) -> str:
    text = text.strip().strip("*`\"'").strip()
    text = text.split("(")[0]  # an echoed gloss: "cut (has a blade...)" -> "cut"
    return " ".join(text.replace("_", " ").lower().rstrip(".,;:").split())


def match_ranked(items: list, options: list, k: int = TOP_K) -> dict:
    """Map a model's ranked answers onto the vocabulary: exact match after normalization (case,
    underscores, stray punctuation/quotes, an echoed gloss), duplicates dropped keeping the first
    (highest-ranked) occurrence, truncated to k. Anything unmatched is kept in "invalid" rather
    than fuzzily mapped, so a short list says the model really didn't produce k valid options."""
    lookup = {_normalize_option(o): o for o in options}
    ranked, invalid = [], []
    for item in items:
        opt = lookup.get(_normalize_option(item))
        if opt is None:
            if item.strip():
                invalid.append(item.strip())
        elif opt not in ranked:
            ranked.append(opt)
    return {"ranked": ranked[:k], "n_valid": len(ranked[:k]), "invalid": invalid}


def shuffled_options(options: list, key: str) -> list:
    """The vocabulary in a fixed random order for one part (seeded by e.g. its region_id), for the
    top-K generation prompt. Listed alphabetically, the 231-word synthia list pulled answers toward
    its first words (Phi-3.5: 96% of answers from the first 20 words; the others 16-28%, vs 9% if
    spread evenly). The same key gives every model the same order for that part, so models stay
    comparable; parsing is unaffected (it matches against the option set)."""
    import random
    import zlib
    order = list(options)
    random.Random(zlib.crc32(key.encode())).shuffle(order)
    return order


def parse_ranked(text: str, options: list, k: int = TOP_K) -> dict:
    """Free-text numbered list -> match_ranked(). Accepts "1. x" / "1) x" / "- x" lines; a single
    line holding a comma-separated list is split on commas instead."""
    import re
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = lines[0].split(",")
    items = [re.sub(r"^\s*(?:\d+\s*[.):-]|[-*•])\s*", "", l) for l in lines]
    return match_ranked(items, options, k)

def build_per_part_question(part_idx: int, n_parts: int, options: list, glosses: dict) -> str:
    listed = ", ".join(f"{o} ({glosses[o]})" if o in glosses else o for o in options)
    # The "answer none" instruction only makes sense when "none" is actually one of the scored
    # options (VOCABULARIES["agd20k"]) -- omitted for a none-less vocabulary (VOCABULARIES
    # ["agd20k_no_none"]) so the prompt doesn't dangle a way out that isn't actually on offer.
    none_clause = f'If part {part_idx} offers none of these, answer "none". ' if "none" in options else ""
    return (
        f"The image shows an object with {n_parts} parts marked and numbered (1 to {n_parts}), each "
        f"outlined and tinted a different color. Focusing specifically on part {part_idx}: consider "
        "whether its affordance differs from the other numbered parts, then choose the single "
        f"affordance part {part_idx} offers from this list: {listed}. {none_clause}"
        "Answer with the option only."
    )


def numbered_overlay(image: Image.Image, masks: list) -> Image.Image:
    """All parts coloured + outlined + numbered at once, as a plain image (no title/axes) for model input."""
    arr = np.array(image).copy()
    overlay = arr.copy()
    for i, m in enumerate(masks):
        overlay[m] = (0.5 * np.array(TAB10[i % 10]) + 0.5 * arr[m]).astype(np.uint8)
    for m in masks:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0), max(2, round(min(m.shape) / 300)))
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4),
                      (255, 255, 255), -1)
        cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th // 2 - 4), (cx + tw // 2 + 4, cy + th // 2 + 4),
                      (0, 0, 0), 1)
        cv2.putText(overlay, label, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return Image.fromarray(overlay)


def first_sentence(text: str) -> str:
    """Model output tends to ramble on well past the requested length; cropping to the first clause/sentence both
    enforces brevity and avoids the mid-word cutoffs a hard token budget alone can produce."""
    text = text.strip().strip('"').lstrip(":").strip()
    cut = min((i for i in (text.find(c) for c in ".!?") if i != -1), default=len(text))
    return text[:cut].strip() if cut < len(text) else text
