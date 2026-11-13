"""
Per-object region-label figures for the 5 affordance-guided novel objects -- one PNG per object,
pairing the numbered-region overlay label_affordance_guided_synthia.py saved (figures/
affordance_guided_novel_objects/sam_regions_partitioned/<scenario_id>_numbered.png -- the
POST-partition_regions() parts that were actually scored, not the raw pre-partition SAM3 candidate
visualization generate_regions_affordance_guided.py saved) with a table of Qwen3-VL-8B's
synthia_filtered label + justification for every part, plus the whole-object question's answer as a
caption below.

Single model this time (unlike plot_object_region_labels.py's 3-model comparison), so the table has
room to show the justification text too, not just the label -- useful here specifically because the
SAM3 regions for these objects are noticeably rougher than the main study's (hand-rolled grid +
NMS, no SLIC/merge cleanup), so seeing WHY a part got a given label makes it much easier to tell a
genuinely bad label apart from a genuinely bad region.

Usage:
    venv_vlm/bin/python3 plot_affordance_guided_region_labels.py             # all 5 scenarios
    venv_vlm/bin/python3 plot_affordance_guided_region_labels.py water_plant # just this one
Output: manuscript_figures/affordance_guided_region_labels/<scenario_id>.png
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # affordance-test
BASE_IMG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "base_images")
IMG_DIR = os.path.join(HERE, "figures", "affordance_guided_novel_objects", "sam_regions_partitioned")
JSON_DIR = os.path.join(HERE, "json", "affordance_guided_objects_justification_qwen3-vl-8b_synthia")
OUT_DIR = os.path.join(HERE, "manuscript_figures", "affordance_guided_region_labels")

SCENARIOS = ["water_plant", "scoop_stir_soup", "support_human_item", "cut_food", "carry_items"]
MODEL_COLOR = "#0072B2"  # same Qwen3-VL-8B color as plot_model_agreement_overview.py / plot_object_region_labels.py
NONE_COLOR = "#999999"
ROW_ALT_BG = "#F4F4F4"
HEADER_BG = "#2B2B2B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#444444",
})


def make_figure(scenario_id: str):
    base_img_path = os.path.join(BASE_IMG_DIR, f"{scenario_id}.png")
    base_image = Image.open(base_img_path).convert("RGB")
    img_path = os.path.join(IMG_DIR, f"{scenario_id}_numbered.png")
    image = Image.open(img_path).convert("RGB")
    with open(os.path.join(JSON_DIR, f"{scenario_id}_affordance_justification.json")) as f:
        data = json.load(f)
    regions = data["regions"]  # already in part order (same order compute_parts() produced them)
    overall = data["overall"]
    n_parts = len(regions)

    row_h = 0.85  # inches per table row -- taller than plot_object_region_labels.py's since
                  # justification text wraps to multiple lines here
    table_h = max(2.0, 0.75 + n_parts * row_h)
    img_aspect = image.height / image.width
    img_w_in = 4.0  # two image panels now (original + regioned), so each is a bit narrower than
                     # the single 5.0in panel plot_object_region_labels.py uses
    img_h_in = img_w_in * img_aspect
    fig_h = max(img_h_in, table_h) + 1.3  # + room for title/caption above and whole-object caption below
    fig_w = img_w_in * 2 + 6.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[img_w_in, img_w_in, 6.5], left=0.03, right=0.98,
                           top=1 - 0.75 / fig_h, bottom=1.05 / fig_h, wspace=0.05)
    ax_base = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[0, 2])

    ax_base.imshow(base_image)
    ax_base.set_xticks([])
    ax_base.set_yticks([])
    ax_base.set_title("Original", fontsize=10, color="#555555", pad=4)
    for s in ax_base.spines.values():
        s.set_visible(False)

    ax_img.imshow(image)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("SAM3 regions", fontsize=10, color="#555555", pad=4)
    for s in ax_img.spines.values():
        s.set_visible(False)

    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)
    ax_tbl.axis("off")

    col_x = {"part": 0.02, "label": 0.10, "justification": 0.43}
    header_y = 1.0
    row_top = header_y - 0.045
    row_h_frac = row_top / max(n_parts, 1)

    ax_tbl.add_patch(FancyBboxPatch((0.0, header_y - 0.04), 1.0, 0.04, boxstyle="square,pad=0",
                                     transform=ax_tbl.transAxes, facecolor=HEADER_BG, edgecolor="none", zorder=1))
    ax_tbl.text(col_x["part"], header_y - 0.02, "Part", transform=ax_tbl.transAxes, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=2)
    ax_tbl.text(col_x["label"], header_y - 0.02, "Qwen3-VL-8B label", transform=ax_tbl.transAxes, ha="left",
                va="center", fontsize=10.5, fontweight="bold", color=MODEL_COLOR, zorder=2)
    ax_tbl.text(col_x["justification"], header_y - 0.02, "Justification", transform=ax_tbl.transAxes, ha="left",
                va="center", fontsize=10.5, fontweight="bold", color="white", zorder=2)

    for i, region in enumerate(regions):
        y_top = row_top - i * row_h_frac
        y_center = y_top - row_h_frac / 2
        if i % 2 == 1:
            ax_tbl.add_patch(FancyBboxPatch((0.0, y_top - row_h_frac), 1.0, row_h_frac, boxstyle="square,pad=0",
                                             transform=ax_tbl.transAxes, facecolor=ROW_ALT_BG, edgecolor="none", zorder=0))
        label = region["vocabularies"]["synthia_filtered"]["label"]
        justification = region["vocabularies"]["synthia_filtered"]["justification"]
        is_none = label.strip().lower() == "none"
        ax_tbl.text(col_x["part"], y_center, str(i + 1), transform=ax_tbl.transAxes, ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color="#222222")
        ax_tbl.text(col_x["label"], y_center, label, transform=ax_tbl.transAxes, ha="left", va="center",
                    fontsize=10, color=NONE_COLOR if is_none else MODEL_COLOR, style="italic" if is_none else "normal")
        wrap_lines = textwrap.wrap(justification, width=52)
        truncated = len(wrap_lines) > 3
        wrapped = "\n".join(wrap_lines[:3])
        if truncated:
            wrapped = wrapped.rstrip() + "…"
        ax_tbl.text(col_x["justification"], y_center, wrapped, transform=ax_tbl.transAxes, ha="left", va="center",
                    fontsize=8.3, color="#444444")

    fig.suptitle(scenario_id.replace("_", " ").title(), fontsize=14, fontweight="bold", x=0.03, ha="left",
                 y=1 - 0.1 / fig_h)
    fig.text(0.03, 1 - 0.35 / fig_h,
              f"{n_parts} SAM3 part{'s' if n_parts != 1 else ''} (post-partition) • synthia_filtered vocabulary, "
              "232 options • italic gray = \"none\"", fontsize=9, color="#555555")

    overall_wrapped = "\n".join(textwrap.wrap(
        f"Whole object: \"{overall['label']}\" — {overall['justification']}", width=140)[:2])
    fig.text(0.03, 0.45 / fig_h, overall_wrapped, fontsize=9.5, color="#111111", fontweight="medium")

    out_png = os.path.join(OUT_DIR, f"{scenario_id}.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenarios = sys.argv[1:] or SCENARIOS
    for scenario_id in scenarios:
        json_path = os.path.join(JSON_DIR, f"{scenario_id}_affordance_justification.json")
        if not os.path.isfile(json_path):
            print(f"  {scenario_id}: no labeling output yet at {json_path}, skipping")
            continue
        out = make_figure(scenario_id)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
