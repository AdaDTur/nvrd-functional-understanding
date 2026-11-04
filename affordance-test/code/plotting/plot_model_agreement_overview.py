"""
Figure 1: overview of how often the 3 VLMs' forced-choice affordance labels agree with each other,
across every region of all 30 novel objects (json/study_manifests/<object>.json).

Three panels:
  (a) pairwise + all-three exact-label agreement rate
  (b) which labels show up most often as a >=2-model consensus pick (including "none", shown
      separately since it dominates and is itself a finding, not a real affordance answer)
  (c) per-object >=2-of-3 agreement rate, sorted -- the all-3-agree rate is near zero almost
      everywhere (1.9% overall) and not informative object-by-object, so this panel uses the
      looser majority bar, which has real spread (0% - 100%).

Output: manuscript_figures/model_agreement_overview.png

Usage: venv_vlm/bin/python3 plot_model_agreement_overview.py
"""
import json
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_CODE_ROOT = os.path.dirname(_SCRIPT_DIR)
for _sub in os.listdir(_CODE_ROOT):
    _p = os.path.join(_CODE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
HERE = os.path.dirname(_CODE_ROOT)  # affordance-test (scripts moved under code/<category>/)
OUT_DIR = os.path.join(HERE, "manuscript_figures")
os.makedirs(OUT_DIR, exist_ok=True)

OBJECTS = ["shape_1", "shape_2", "shape_3", "shape_10", "shape_11", "shape_12", "shape_13",
           "shape_14", "shape_15", "shape_16", "shape_17", "shape_18", "shape_19", "shape_20",
           "shape_21", "shape_22", "shape_23", "shape_24", "shape_25", "shape_26", "shape_27",
           "shape_28", "shape_29", "shape_30", "shape_31", "shape_32", "shape_33", "shape_38",
           "shape_39", "shape_40"]
MODELS = ["qwen3-vl-8b", "internvl3-8b", "molmo2-8b"]
MODEL_DISPLAY = {"qwen3-vl-8b": "Qwen3-VL-8B", "internvl3-8b": "InternVL3-8B", "molmo2-8b": "Molmo2-8B"}

# Okabe-Ito colorblind-safe palette -- same model->color mapping used in the per-object part-label
# figures (plot_object_region_labels.py), so the two figure types read as one consistent system.
MODEL_COLOR = {"qwen3-vl-8b": "#0072B2", "internvl3-8b": "#D55E00", "molmo2-8b": "#009E73"}
NEUTRAL = "#6B6B6B"
NONE_COLOR = "#B0B0B0"
ACCENT = "#0072B2"
GRID_COLOR = "#E3E3E3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.5,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
})


def load_regions():
    regions = []
    for obj in OBJECTS:
        path = os.path.join(HERE, "json", "study_manifests", f"{obj}.json")
        with open(path) as f:
            data = json.load(f)
        for r in data["regions"]:
            labels = {m: r["labels"][m].strip().lower() for m in MODELS}
            regions.append((obj, r["part"], labels))
    return regions


def majority_label(labels: dict):
    vals = list(labels.values())
    for v in set(vals):
        if vals.count(v) >= 2:
            return v
    return None


def main():
    regions = load_regions()
    n_total = len(regions)

    # --- panel (a): pairwise + all-three agreement ---
    pair_agree, pair_total = Counter(), Counter()
    all3 = 0
    for _, _, labels in regions:
        for m1, m2 in combinations(MODELS, 2):
            pair_total[(m1, m2)] += 1
            if labels[m1] == labels[m2]:
                pair_agree[(m1, m2)] += 1
        if len({labels[m] for m in MODELS}) == 1:
            all3 += 1

    pair_order = [("qwen3-vl-8b", "internvl3-8b"), ("qwen3-vl-8b", "molmo2-8b"),
                  ("internvl3-8b", "molmo2-8b")]
    a_labels = [f"{MODEL_DISPLAY[m1]}\n↔ {MODEL_DISPLAY[m2]}" for m1, m2 in pair_order] + ["All three\nagree"]
    a_values = [pair_agree[p] / pair_total[p] for p in pair_order] + [all3 / n_total]

    # --- panel (b): top consensus labels (>=2 models agree on this exact word) ---
    label_consensus = Counter()
    for _, _, labels in regions:
        lab = majority_label(labels)
        if lab is not None:
            label_consensus[lab] += 1
    top_labels = label_consensus.most_common(10)[::-1]  # reversed so highest ends up on top in barh

    # --- panel (c): per-object majority (>=2-of-3) agreement rate ---
    obj_rate = {}
    for obj in OBJECTS:
        regs = [labels for o, _, labels in regions if o == obj]
        n_maj = sum(1 for labels in regs if majority_label(labels) is not None)
        obj_rate[obj] = n_maj / len(regs) if regs else 0.0
    obj_sorted = sorted(obj_rate.items(), key=lambda kv: kv[1])  # ascending so best is at top of barh

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.75], hspace=0.22, wspace=0.38,
                           left=0.09, right=0.97, top=0.91, bottom=0.095)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    fig.suptitle("Agreement among the three VLMs' forced-choice affordance labels",
                 fontsize=14.5, fontweight="bold", x=0.09, ha="left", y=0.965)
    fig.text(0.09, 0.935,
              f"{n_total} regions across {len(OBJECTS)} novel objects • 36 AGD20K-style verbs + \"none\"",
              fontsize=10, color="#555555")

    # (a) pairwise + all-3
    bar_colors = [NEUTRAL, NEUTRAL, NEUTRAL, ACCENT]
    bars = ax_a.bar(range(len(a_labels)), a_values, color=bar_colors, width=0.6, zorder=3)
    ax_a.set_xticks(range(len(a_labels)))
    ax_a.set_xticklabels(a_labels, fontsize=9.3)
    ax_a.set_ylabel("Agreement rate")
    ax_a.set_ylim(0, max(a_values) * 1.35)
    ax_a.set_title("(a)  Exact-label agreement", fontsize=11, fontweight="bold", loc="left")
    ax_a.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_a.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_a.spines[spine].set_visible(False)
    for b, v in zip(bars, a_values):
        ax_a.text(b.get_x() + b.get_width() / 2, v + max(a_values) * 0.03, f"{v:.0%}",
                   ha="center", va="bottom", fontsize=9.3, fontweight="medium")

    # (b) top consensus labels
    names = [n for n, _ in top_labels]
    counts = [c for _, c in top_labels]
    colors = [NONE_COLOR if n == "none" else ACCENT for n in names]
    ax_b.barh(range(len(names)), counts, color=colors, height=0.62, zorder=3)
    ax_b.set_yticks(range(len(names)))
    ax_b.set_yticklabels([n if n != "none" else "none (abstained)" for n in names], fontsize=9.6)
    ax_b.set_xlabel("Regions where ≥ 2 models agreed on this label")
    ax_b.set_title("(b)  Most common points of consensus", fontsize=11, fontweight="bold", loc="left")
    ax_b.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)
    for i, c in enumerate(counts):
        ax_b.text(c + max(counts) * 0.02, i, str(c), va="center", fontsize=9, color="#333333")

    # (c) per-object majority agreement
    obj_names = [o for o, _ in obj_sorted]
    obj_vals = [v for _, v in obj_sorted]
    ax_c.barh(range(len(obj_names)), obj_vals, color=ACCENT, height=0.68, zorder=3)
    ax_c.set_yticks(range(len(obj_names)))
    ax_c.set_yticklabels(obj_names, fontsize=8.6)
    ax_c.set_xlabel("Share of regions where ≥ 2 of 3 models picked the same label")
    ax_c.set_xlim(0, 1.0)
    ax_c.set_title("(c)  Per-object agreement, ≥ 2-of-3 models (sorted)", fontsize=11, fontweight="bold", loc="left")
    ax_c.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_c.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_c.spines[spine].set_visible(False)
    for i, v in enumerate(obj_vals):
        if v > 0:
            ax_c.text(v + 0.012, i, f"{v:.0%}", va="center", fontsize=7.6, color="#333333")

    fig.text(0.09, 0.015,
              "Pilot error analysis — UIUC novel-concept affordance study. "
              "“None (abstained)” counted as a label like any other.",
              fontsize=8, color="#777777")

    out_png = os.path.join(OUT_DIR, "model_agreement_overview.png")
    fig.savefig(out_png, dpi=300)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
