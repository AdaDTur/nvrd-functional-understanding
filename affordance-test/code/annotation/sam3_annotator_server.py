"""
Interactive click-to-segment annotator for the affordance-guided novel objects, backed by live
SAM3 inference (model.predict_inst(), the same point-promptable interface run_sam3_test.py and
generate_regions_affordance_guided.py's automatic grid already use -- here driven by manual clicks
instead of a dense grid).

Why: generate_regions_affordance_guided.py's automatic SAM3 regions are noticeably rough for
several of these objects (confirmed visually in the region-label figures -- see carry_items'
sliver parts). This tool lets a person directly point at the part they mean and get SAM3's best
mask for that click, accepting or refining it before moving to the next region -- a manual
correction pass, not a different automatic method.

Architecture: SAM3 needs a live GPU-backed Python process, so this is a real server (Flask), not a
static page -- it loads SAM3 ONCE at startup and keeps it resident, then answers click requests
over HTTP. Access it by opening http://localhost:5050 (or the forwarded port) in a browser; if
you're on a remote/SSH connection to this machine, forward port 5050 first (VS Code does this
automatically for a port a process is listening on).

Per-object interaction model -- two ways to prompt a region, either can be committed the same way:
  - Point mode: click = positive point (this IS part of the region), shift+click = negative point
    (this is NOT part of the region) -- same convention as SAM's own interactive demo. Each click
    re-runs predict_inst() with ALL accumulated points.
  - Outline mode: click-and-drag a rough freehand loop around the part. On mouse-up, the loop is
    rasterized to a filled mask and combined with its bounding box as a box+mask_input prompt --
    the same SAM1-task refinement interface run_sam3_test.py's Test 2 verified (there, refining an
    existing region; here, a user's rough freehand shape), which snaps the rough shape to the
    model's own sense of the real part boundary rather than just filling the loop as drawn.
  - "Commit region" finalizes whichever preview mask is currently showing (from either mode) along
    with a typed-in ground-truth affordance label, as a saved region, and clears the in-progress
    state. The label field autocompletes against the exact same synthia_filtered vocabulary
    (synthia_affordances_clustered.txt) label_affordance_guided_synthia.py scores against, so the
    stored ground truth and the models' own labels are directly comparable word-for-word -- free
    text is still allowed (the field doesn't force a choice), for a region that genuinely needs a
    word outside that list.
  - "Save" writes every committed region for that object to
    json/affordance_guided_objects_regions_manual/<object_id>.json, in the exact region_id/area/
    segmentation-as-COCO-RLE schema the rest of the pipeline already reads (plus a "ground_truth"
    field per region) -- a SEPARATE directory from the automatic
    json/affordance_guided_objects_regions/, so neither overwrites the other and the
    automatic-vs-manual regions stay directly comparable.

Usage:
    venv_sam3/bin/python3 sam3_annotator_server.py
"""
import base64
import io
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # SAM3's GPU path isn't reliable here -- set_image()'s own
# forward pass runs outside any autocast context and crashes on a bf16/float32 mismatch against the
# model's bf16 weights (confirmed directly); every other SAM3 usage in this project stays on CPU
# for the same reason (see generate_regions_affordance_guided.py's module docstring), so this
# interactive tool does too, favoring reliability over per-click latency.

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image
from pycocotools import mask as mask_utils

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test

sys.path.insert(0, os.path.join(HERE, "sam2_experiment", "sam3"))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

IMG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "base_images")
OUT_DIR = os.path.join(HERE, "json", "affordance_guided_objects_regions_manual")
os.makedirs(OUT_DIR, exist_ok=True)

# Same vocabulary source VOCABULARIES["synthia_filtered"] loads from in affordance_labeling.py --
# read directly here (not imported from there) since that module pulls in cv2/transformers, not
# installed in venv_sam3. Ground-truth labels autocomplete against this so they're directly
# comparable to label_affordance_guided_synthia.py's own output, word-for-word.
with open(os.path.join(HERE, "synthia_affordances_clustered.txt")) as _f:
    GT_VOCAB = [line.strip() for line in _f if line.strip()] + ["none"]

PORT = 5050

# TAB10-style palette, reused from affordance_labeling.TAB10 (hardcoded here, not imported --
# that module pulls in cv2/transformers which aren't installed in venv_sam3).
PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
PREVIEW_COLOR = (255, 0, 220)  # in-progress (uncommitted) mask, same magenta HIGHLIGHT_INTRO uses elsewhere

# SAM3's image size is 1008px with a 72x72 lowest-res feature grid (1008/14), so mask_input needs
# to be 72*4=288x288 -- confirmed directly in run_sam3_test.py (a hardcoded 256, copied from SAM1/
# SAM2's convention, produced a shape-mismatch error there: "72 must match 64").
MASK_INPUT_SIZE = 288

app = Flask(__name__)

DEVICE = "cpu"
print("Loading SAM3 on cpu (downloads the checkpoint from HF on first run)...")
MODEL = build_sam3_image_model(device=DEVICE, enable_inst_interactivity=True)
MODEL = MODEL.float()  # bf16 tensors crash plain CPU matmuls without autocast
PROCESSOR = Sam3Processor(MODEL, device=DEVICE)
print("SAM3 ready.")

STATE = {}  # object_id -> {image, sam_state, points: [(x,y,label)], regions: [{"id","mask"}], next_id}


def predict_from_points(sam_state, points: list):
    """points: [(x, y, label), ...]. Returns (mask: bool ndarray, score: float)."""
    point_coords = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
    point_labels = np.array([l for _, _, l in points], dtype=np.int64)
    masks, scores, _ = MODEL.predict_inst(
        sam_state, point_coords=point_coords, point_labels=point_labels, multimask_output=True
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def predict_from_outline(sam_state, path: list, img_w: int, img_h: int):
    """path: [(x, y), ...] in original image pixel coords -- a closed rough freehand loop. Fills it
    to a mask, converts to SAM3's mask_input logit convention, and combines with the loop's own
    bounding box as a box+mask_input prompt (the exact refinement call shape run_sam3_test.py's
    Test 2 already proved out). Returns (mask: bool ndarray, score: float)."""
    from PIL import ImageDraw
    rough = Image.new("L", (img_w, img_h), 0)
    ImageDraw.Draw(rough).polygon(path, fill=255)
    rough_arr = np.array(rough)

    xs, ys = [p[0] for p in path], [p[1] for p in path]
    box = np.array([max(0, min(xs)), max(0, min(ys)), min(img_w - 1, max(xs)), min(img_h - 1, max(ys))],
                    dtype=np.float32)

    mask_resized = np.array(Image.fromarray(rough_arr).resize((MASK_INPUT_SIZE, MASK_INPUT_SIZE), Image.NEAREST))
    mask_input = np.where(mask_resized > 127, 8.0, -8.0).astype(np.float32)[None, :, :]

    masks, scores, _ = MODEL.predict_inst(sam_state, box=box[None, :], mask_input=mask_input, multimask_output=False)
    return masks[0].astype(bool), float(scores[0])


def get_state(object_id: str) -> dict:
    if object_id not in STATE:
        img_path = os.path.join(IMG_DIR, f"{object_id}.png")
        image = Image.open(img_path).convert("RGB")
        sam_state = PROCESSOR.set_image(image)
        STATE[object_id] = {
            "image": image, "sam_state": sam_state, "points": [],
            "pending_mask": None, "pending_score": None, "regions": [], "next_id": 1,
        }
    return STATE[object_id]


def mask_to_png_b64(mask: np.ndarray, color: tuple, alpha: int = 130) -> str:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, alpha)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def object_list() -> list:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith(".png"))


def state_json(object_id: str, st: dict) -> dict:
    preview_mask = st["pending_mask"]
    return {
        "object_id": object_id,
        "width": st["image"].width,
        "height": st["image"].height,
        "points": [{"x": x, "y": y, "label": l} for x, y, l in st["points"]],
        "preview": {
            "mask_png": mask_to_png_b64(preview_mask, PREVIEW_COLOR) if preview_mask is not None else None,
            "score": st["pending_score"],
        },
        "regions": [
            {"id": r["id"], "area": int(r["mask"].sum()), "ground_truth": r["ground_truth"],
             "mask_png": mask_to_png_b64(r["mask"], PALETTE[i % len(PALETTE)])}
            for i, r in enumerate(st["regions"])
        ],
    }


@app.route("/")
def index():
    return send_file(os.path.join(_SCRIPT_DIR, "sam3_annotator.html"))


@app.route("/api/objects")
def api_objects():
    return jsonify(object_list())


@app.route("/api/vocab")
def api_vocab():
    return jsonify(GT_VOCAB)


@app.route("/api/object/<object_id>/image")
def api_image(object_id):
    return send_file(os.path.join(IMG_DIR, f"{object_id}.png"))


@app.route("/api/object/<object_id>/state")
def api_state(object_id):
    st = get_state(object_id)
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/point", methods=["POST"])
def api_point(object_id):
    st = get_state(object_id)
    data = request.json
    st["points"].append((float(data["x"]), float(data["y"]), int(data["label"])))
    st["pending_mask"], st["pending_score"] = predict_from_points(st["sam_state"], st["points"])
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/outline", methods=["POST"])
def api_outline(object_id):
    st = get_state(object_id)
    data = request.json
    path = [(float(p["x"]), float(p["y"])) for p in data["points"]]
    if len(path) < 3:
        return jsonify({"error": "outline needs at least 3 points"}), 400
    st["points"] = []  # outline mode replaces any in-progress point prompt, not combined with it
    st["pending_mask"], st["pending_score"] = predict_from_outline(
        st["sam_state"], path, st["image"].width, st["image"].height
    )
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/undo_point", methods=["POST"])
def api_undo_point(object_id):
    st = get_state(object_id)
    if st["points"]:
        st["points"].pop()
    if st["points"]:
        st["pending_mask"], st["pending_score"] = predict_from_points(st["sam_state"], st["points"])
    else:
        st["pending_mask"], st["pending_score"] = None, None
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/clear_points", methods=["POST"])
def api_clear_points(object_id):
    st = get_state(object_id)
    st["points"] = []
    st["pending_mask"], st["pending_score"] = None, None
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/commit", methods=["POST"])
def api_commit(object_id):
    st = get_state(object_id)
    if st["pending_mask"] is None:
        return jsonify({"error": "no in-progress region to commit"}), 400
    ground_truth = (request.json or {}).get("ground_truth", "").strip()
    if not ground_truth:
        return jsonify({"error": "enter a ground-truth affordance label before committing"}), 400
    region_id = f"{object_id}_manual_{st['next_id']}"
    st["regions"].append({"id": region_id, "mask": st["pending_mask"], "ground_truth": ground_truth})
    st["next_id"] += 1
    st["points"] = []
    st["pending_mask"], st["pending_score"] = None, None
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/delete_region/<region_id>", methods=["POST"])
def api_delete_region(object_id, region_id):
    st = get_state(object_id)
    st["regions"] = [r for r in st["regions"] if r["id"] != region_id]
    return jsonify(state_json(object_id, st))


@app.route("/api/object/<object_id>/save", methods=["POST"])
def api_save(object_id):
    st = get_state(object_id)
    regions = []
    for r in st["regions"]:
        rle = mask_utils.encode(np.asfortranarray(r["mask"].astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        regions.append({"region_id": r["id"], "area": int(r["mask"].sum()), "segmentation": rle,
                         "ground_truth": r["ground_truth"]})
    out_path = os.path.join(OUT_DIR, f"{object_id}.json")
    with open(out_path, "w") as f:
        json.dump(regions, f)
    return jsonify({"saved": out_path, "n_regions": len(regions)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
