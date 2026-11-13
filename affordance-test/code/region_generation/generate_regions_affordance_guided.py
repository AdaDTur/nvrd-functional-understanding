"""
Region generation for the affordance-guided novel objects
(figures/affordance_guided_novel_objects/base_images/<scenario_id>.png -- the openai/gpt-image-2
generations that passed the generate+judge loop in image_generation/generate_affordance_guided.py), via SAM2
automatic mask generation -- same SAM2AutomaticMaskGenerator, same points_per_side=32/crop_n_layers=1
settings, as the main 30-object study's generate_regions_sam2.py.

This REPLACES an earlier SAM3-based version (a hand-rolled grid-of-points + greedy-IoU-NMS dedup,
since SAM3 ships no AutomaticMaskGenerator of its own). That version was picked after SAM3 looked
cleaner on ONE object (carry_items' large cavities) in a 5-object comparison, but at 25+ objects the
hand-rolled approximation turned out choppy and inconsistent -- a crude point-grid substitute is a
weaker foundation than SAM2's purpose-built, stability-filtered automatic generator, and a single
favorable example didn't generalize. SAM3 is still used elsewhere in this project, just not for
automatic coverage here: the manual annotator (code/annotation/sam3_annotator_server.py) keeps using
SAM3's point/box-prompted interactive mode, which is a genuinely different, better-suited use of it
(user-directed refinement, not blind automatic generation).

Usage:
    venv_vlm/bin/python3 generate_regions_affordance_guided.py              # every base image present
    venv_vlm/bin/python3 generate_regions_affordance_guided.py water_plant  # just this one

Output (same region_id/area/segmentation-as-COCO-RLE schema load_regions()/partition_regions()
already expect elsewhere in this pipeline, so these are usable downstream unmodified):
    json/affordance_guided_objects_regions/<scenario_id>.json
Plus a numbered visualization of the raw (pre-partition_regions) candidate masks for a quick visual
sanity check:
    figures/affordance_guided_novel_objects/sam_regions/<scenario_id>_numbered.png
"""
import colorsys
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test

from gpu_contention import wait_for_gpu_headroom

IMG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "base_images")
JSON_OUT_DIR = os.path.join(HERE, "json", "affordance_guided_objects_regions")
FIG_OUT_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "sam_regions")

SAM2_ROOT = os.path.join(HERE, "sam2_experiment", "sam2")
CHECKPOINT = os.path.join(HERE, "sam2_experiment", "checkpoints", "sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def available_scenarios() -> list:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith(".png"))


def render_numbered(image, masks_list, out_path, max_masks=40):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    masks_list = sorted(masks_list, key=lambda m: -m["area"])[:max_masks]
    canvas = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    n = len(masks_list)
    for i, m in enumerate(masks_list):
        hue = (i / max(n, 1)) % 1.0
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.6, 0.95)]
        seg = m["segmentation"]
        mask_img = Image.fromarray((seg * 255).astype(np.uint8))
        color_layer = Image.new("RGBA", canvas.size, (r, g, b, 110))
        overlay.paste(color_layer, (0, 0), mask_img)
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, m in enumerate(masks_list, start=1):
        ys, xs = np.nonzero(m["segmentation"])
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        draw.text((cx, cy), str(i), fill=(0, 0, 0), font=font, anchor="mm")
    canvas.save(out_path)
    return len(masks_list)


def save_regions(scenario_id: str, masks_list: list, out_dir: str) -> str:
    import numpy as np
    from pycocotools import mask as mask_utils

    regions = []
    for i, m in enumerate(masks_list):
        seg = m["segmentation"]
        rle = mask_utils.encode(np.asfortranarray(seg.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        regions.append({"region_id": f"{scenario_id}_region_{i}", "area": int(m["area"]), "segmentation": rle})
    out_path = os.path.join(out_dir, f"{scenario_id}.json")
    with open(out_path, "w") as f:
        json.dump(regions, f)
    return out_path


def main():
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, SAM2_ROOT)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    scenarios = sys.argv[1:] or available_scenarios()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    wait_for_gpu_headroom(log=print)
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, points_per_side=32, crop_n_layers=1)

    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_OUT_DIR, exist_ok=True)

    for scenario_id in scenarios:
        wait_for_gpu_headroom(log=print)
        img_path = os.path.join(IMG_DIR, f"{scenario_id}.png")
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            masks = mask_generator.generate(image_np)
        masks_list = [{"segmentation": m["segmentation"], "area": int(m["segmentation"].sum())} for m in masks]
        out_path = save_regions(scenario_id, masks_list, JSON_OUT_DIR)
        n = render_numbered(image, masks_list, os.path.join(FIG_OUT_DIR, f"{scenario_id}_numbered.png"))
        print(f"{scenario_id}: {len(masks_list)} candidate masks -> {out_path} ({n} shown in overlay)")


if __name__ == "__main__":
    main()
