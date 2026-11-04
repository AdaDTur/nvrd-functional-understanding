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
    cache-free full-forward reference in verify_methods.py.

Images are downscaled to MAX_SIDE px: the shared GPUs are often too full for full-resolution vision-tower
activations, and answers are somewhat resolution-sensitive, so compare runs at equal MAX_SIDE.

Entry points live in other files (batch_final_labeling.py etc.); this module has no CLI and parses no argv.
"""

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


def build_prompt_text(processor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def _supports_cache_position(model) -> bool:
    """Whether model.forward() accepts cache_position -- most do, but a custom (trust_remote_code)
    forward() isn't guaranteed to: kimi-vl-a3b's forward() raised a hard TypeError on it ("unexpected
    keyword argument"), confirming its signature has neither that named parameter nor a **kwargs
    catch-all. Checked once per model rather than hardcoding a list of which models lack it, so this
    generalizes to whatever gets added to MODELS next without another signature-mismatch surprise."""
    params = inspect.signature(model.forward).parameters
    return "cache_position" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


@torch.no_grad()
def score_options(model, processor, image: Image.Image, prompt_text: str, option_ids: list, device) -> np.ndarray:
    """log P(option + <|im_end|> | prompt, image) for every option. The image+prompt prefix is run
    once; each option's continuation reuses that KV cache, which is cropped back afterwards."""
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    cache = DynamicCache()
    out = model(**inputs, past_key_values=cache, use_cache=True)
    first_logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    use_cache_position = _supports_cache_position(model)

    scores = []
    for ids in option_ids:
        lp = first_logp[ids[0]].item()
        if len(ids) > 1:
            cont = torch.tensor([ids[:-1]], device=device)
            n = cont.shape[1]
            kwargs = dict(
                input_ids=cont,
                attention_mask=torch.ones(1, prompt_len + n, dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
            )
            if use_cache_position:
                kwargs["cache_position"] = torch.arange(prompt_len, prompt_len + n, device=device)
            o = model(**kwargs)
            logp = F.log_softmax(o.logits[0].float(), dim=-1)
            lp += sum(logp[j, ids[j + 1]].item() for j in range(n))
            cache.crop(prompt_len)
        scores.append(lp)
    return np.array(scores)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

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
