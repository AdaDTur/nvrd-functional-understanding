"""
Region generation for the 5 affordance-guided novel objects
(figures/affordance_guided_novel_objects/base_images/<scenario_id>.png -- the openai/gpt-image-2 generations
that passed the generate+judge loop in generate_affordance_edits_openai.py), via SAM3 automatic
mask generation.

SAM3 ships no AutomaticMaskGenerator class the way SAM2 does, so this is a hand-rolled
grid-of-points + greedy-IoU-NMS dedup (built on model.predict_inst(), the same promptable interface
run_sam3_test.py verified working), mirroring sam2_experiment/run_model_comparison.py's run_sam3().
Kept on CPU, matching every other SAM3 usage in this project -- SAM3 has never actually been run on
GPU here, and the bf16/autocast handling in model_builder.py was only ever exercised on the CPU
path, so moving it to GPU for 5 images isn't worth risking the same kind of compatibility issues
SAM3 already needed patches for elsewhere.

A SAM2 path existed here too (SAM2AutomaticMaskGenerator, mirroring generate_regions_sam2.py) but
was removed: on a direct comparison on these 5 objects, SAM3 gave cleaner, more coherent regions on
the large functional cavities these affordance-guided objects have (carry_items: SAM2 fragmented
the main pockets into overlapping sub-pieces; SAM3 kept each as one coherent mask). This is the
OPPOSITE of what generate_regions_sam2.py's own docstring found on the original abstract shape_N
objects (SAM2 beat SAM3 there, on fine mechanical/textured detail) -- the two object styles suit
different SAM generations, not a universal winner. SAM2 was NOT removed from generate_regions_sam2.py
or the main 30-object study pipeline, which is unrelated and still SAM2-based.

Needs the SAM3 venv (Python 3.12, different from the rest of this pipeline's venv_vlm):
    venv_sam3/bin/python3 generate_regions_affordance_guided.py

Output (same region_id/area/segmentation-as-COCO-RLE schema load_regions()/partition_regions()
already expect elsewhere in this pipeline, so these are usable downstream unmodified):
    json/affordance_guided_objects_regions/<scenario_id>.json
Plus a numbered visualization of the raw (pre-partition_regions) candidate masks for a quick visual
sanity check, same idea as render_numbered() in run_model_comparison.py:
    figures/affordance_guided_novel_objects/sam_regions/<scenario_id>_numbered.png
"""
import colorsys
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # see module docstring -- SAM3's GPU path is untested here

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
IMG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "base_images")
JSON_OUT_DIR = os.path.join(HERE, "json", "affordance_guided_objects_regions")
FIG_OUT_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "sam_regions")

SCENARIOS = ["water_plant", "scoop_stir_soup", "support_human_item", "cut_food", "carry_items"]

GRID_N = 16  # denser than run_model_comparison.py's 12 -- only 5 images here, not a 4-object smoke
             # test, so the extra CPU time is affordable
MARGIN_FRAC = 0.08


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


def mask_iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


def main():
    import numpy as np
    from PIL import Image

    sam3_root = os.path.join(HERE, "sam2_experiment", "sam3")
    sys.path.insert(0, sam3_root)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("device: cpu")
    print("building SAM3 model (downloads the checkpoint from HF on first run)...")
    model = build_sam3_image_model(device="cpu", enable_inst_interactivity=True)
    model = model.float()  # bf16 tensors in the checkpoint crash plain CPU matmuls without autocast
    processor = Sam3Processor(model, device="cpu")

    os.makedirs(JSON_OUT_DIR, exist_ok=True)
    os.makedirs(FIG_OUT_DIR, exist_ok=True)

    for scenario_id in SCENARIOS:
        img_path = os.path.join(IMG_DIR, f"{scenario_id}.png")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        state = processor.set_image(image)

        xs = np.linspace(w * MARGIN_FRAC, w * (1 - MARGIN_FRAC), GRID_N)
        ys = np.linspace(h * MARGIN_FRAC, h * (1 - MARGIN_FRAC), GRID_N)
        candidates = []
        for y in ys:
            for x in xs:
                point_coords = np.array([[x, y]])
                point_labels = np.array([1])
                masks, scores, _ = model.predict_inst(
                    state, point_coords=point_coords, point_labels=point_labels, multimask_output=True
                )
                best = int(np.argmax(scores))
                m = masks[best].astype(bool)
                s = float(scores[best])
                area = int(m.sum())
                if area < 0.0005 * w * h or s < 0.7:
                    continue
                candidates.append((s, area, m))

        # greedy NMS: highest score first, drop anything with high IoU overlap against an
        # already-kept mask (duplicate point hits on the same part are common with a dense grid)
        candidates.sort(key=lambda t: -t[0])
        kept = []
        for s, area, m in candidates:
            if all(mask_iou(m, km) < 0.6 for _, _, km in kept):
                kept.append((s, area, m))

        masks_list = [{"segmentation": m, "area": area} for s, area, m in kept]
        out_path = save_regions(scenario_id, masks_list, JSON_OUT_DIR)
        n = render_numbered(image, masks_list, os.path.join(FIG_OUT_DIR, f"{scenario_id}_numbered.png"))
        print(f"{scenario_id}: {len(candidates)} candidates -> {len(kept)} after NMS -> {out_path} ({n} shown)")


if __name__ == "__main__":
    main()
