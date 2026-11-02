"""
Figure 2: per-object region-label figures for the manuscript supplement -- one PNG/PDF per novel
object, pairing its existing numbered-region overlay (figures/study_images/<object>_numbered.png,
already outlined/tinted/numbered per region by numbered_overlay() in the main pipeline) with a
color-coded table of which label each of the 3 VLMs picked for every numbered part.

Color (not per-part text labels naming the model) carries model identity, consistently across every
object's figure and matching plot_model_agreement_overview.py's palette -- a reader only needs to
learn the 3-color legend once. A table (rather than placing text directly on the photo) was chosen
over on-image leader lines specifically because part count ranges from 1 to 15 across these 30
objects; on-image labels would be legible for the small objects and unreadable clutter for the large
ones, which also means no two objects' figures would share a consistent style -- bad for a
supplementary figure set a reader flips through together.

Output: manuscript_figures/region_labels/<object>_region_labels.png

Usage:
    venv_vlm/bin/python3 plot_object_region_labels.py                   # all 30 novel objects
    venv_vlm/bin/python3 plot_object_region_labels.py shape_14 shape_27 # just these
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(HERE, "figures", "study_images")
MANIFEST_DIR = os.path.join(HERE, "json", "study_manifests")
OUT_DIR = os.path.join(HERE, "manuscript_figures", "region_labels")
os.makedirs(OUT_DIR, exist_ok=True)

OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
           "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
           "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
           "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
           "shape_39", "shape_40"]
MODELS = ["qwen3-vl-8b", "internvl3-8b", "molmo2-8b"]
MODEL_DISPLAY = {"qwen3-vl-8b": "Qwen3-VL-8B", "internvl3-8b": "InternVL3-8B", "molmo2-8b": "Molmo2-8B"}

# Same Okabe-Ito colorblind-safe palette as plot_model_agreement_overview.py.
MODEL_COLOR = {"qwen3-vl-8b": "#0072B2", "internvl3-8b": "#D55E00", "molmo2-8b": "#009E73"}
NONE_COLOR = "#999999"
ROW_ALT_BG = "#F4F4F4"
HEADER_BG = "#2B2B2B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#444444",
})


def load_manifest(obj: str) -> list:
    with open(os.path.join(MANIFEST_DIR, f"{obj}.json")) as f:
        data = json.load(f)
    return sorted(data["regions"], key=lambda r: r["part"])


def make_figure(obj: str):
    img_path = os.path.join(IMG_DIR, f"{obj}_numbered.png")
    image = Image.open(img_path).convert("RGB")
    regions = load_manifest(obj)
    n_parts = len(regions)

    row_h = 0.42  # inches per table row, used to size the figure to the part count
    table_h = max(2.0, 0.75 + n_parts * row_h)
    img_aspect = image.height / image.width
    img_w_in = 5.0
    img_h_in = img_w_in * img_aspect
    fig_h = max(img_h_in, table_h) + 0.9  # + room for title/legend above
    fig_w = img_w_in + 5.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[img_w_in, 5.6], left=0.03, right=0.98,
                           top=1 - 0.85 / fig_h, bottom=0.08, wspace=0.08)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    # --- left: the existing numbered region overlay, unmodified ---
    ax_img.imshow(image)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for s in ax_img.spines.values():
        s.set_visible(False)

    # --- right: per-part label table, color = model identity ---
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)
    ax_tbl.axis("off")

    col_x = {"part": 0.03, "qwen3-vl-8b": 0.17, "internvl3-8b": 0.47, "molmo2-8b": 0.77}
    header_y = 1.0
    row_top = header_y - 0.09
    row_h_frac = row_top / max(n_parts, 1)

    # column headers
    ax_tbl.add_patch(FancyBboxPatch((0.0, header_y - 0.075), 1.0, 0.075, boxstyle="square,pad=0",
                                     transform=ax_tbl.transAxes, facecolor=HEADER_BG,
                                     edgecolor="none", zorder=1))
    ax_tbl.text(col_x["part"], header_y - 0.038, "Part", transform=ax_tbl.transAxes,
                ha="left", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=2)
    for m in MODELS:
        ax_tbl.text(col_x[m], header_y - 0.038, MODEL_DISPLAY[m], transform=ax_tbl.transAxes,
                    ha="left", va="center", fontsize=10.5, fontweight="bold", color=MODEL_COLOR[m],
                    zorder=2)

    for i, region in enumerate(regions):
        y_top = row_top - i * row_h_frac
        y_center = y_top - row_h_frac / 2
        if i % 2 == 1:
            ax_tbl.add_patch(FancyBboxPatch((0.0, y_top - row_h_frac), 1.0, row_h_frac,
                                             boxstyle="square,pad=0", transform=ax_tbl.transAxes,
                                             facecolor=ROW_ALT_BG, edgecolor="none", zorder=0))
        ax_tbl.text(col_x["part"], y_center, str(region["part"]), transform=ax_tbl.transAxes,
                    ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222222")
        for m in MODELS:
            label = region["labels"][m].strip().lower()
            is_none = label == "none"
            ax_tbl.text(col_x[m], y_center, label, transform=ax_tbl.transAxes,
                        ha="left", va="center", fontsize=10,
                        color=NONE_COLOR if is_none else MODEL_COLOR[m],
                        style="italic" if is_none else "normal")

    fig.suptitle(obj.replace("_", " ").title(), fontsize=14, fontweight="bold", x=0.03, ha="left",
                 y=1 - 0.08 / fig_h)
    fig.text(0.03, 1 - 0.33 / fig_h,
              f"{n_parts} region{'s' if n_parts != 1 else ''} • forced-choice label per model "
              "(italic gray = \"none\")", fontsize=9, color="#555555")

    out_png = os.path.join(OUT_DIR, f"{obj}_region_labels.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def main():
    objects = sys.argv[1:] or OBJECTS
    for obj in objects:
        out = make_figure(obj)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
