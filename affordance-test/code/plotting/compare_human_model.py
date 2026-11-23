"""
Human vs model functionality judgments on the 30 original novel objects (synthia vocabulary).

Human data: responses_long.csv (from code/study_assets/pull_responses.py), one row per participant x
answer. Model data: json/affordance_justification_<model>_synthia/<obj>_affordance_justification.json
(top-10 lists per part: "generation" for all five models, "logprob" for the four open models).

Every comparison is per part, scored against one held-out human at a time:
  hit@k        P(at least one of the human's picks is in the model's top-k), k = 1..10
  top-3 Jaccard  |model top-3 & human picks| / |model top-3 | human picks|
The HUMAN CEILING uses the same scoring with the model's list replaced by the OTHER participants' picks
on that part, ranked by how many of them chose each label (leave-one-out crowd). It needs >= 2 humans
on a part, so all metrics are computed on those parts only, for models and humans alike.

Caveats printed with the results:
  - The options humans clicked were the Borda top-10 of Molmo2, Gemma-4, Qwen3.5 and GPT-5.4's generation
    lists, so agreement with those four is inflated relative to Phi-3.5 and to the log-prob mode.
  - Phi-3.5's synthia generation lists were never re-run with the shuffled option order (the others were),
    so they still carry the alphabetical-position bias.

Outputs: figures/human_vs_model/*.png and json/human_model_comparison.json
Usage (from nvrd/affordance-test/): ../venv_vlm/bin/python3 code/plotting/compare_human_model.py
"""
import collections
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # affordance-test
OUT_FIG = os.path.join(HERE, "figures", "human_vs_model")
OUT_JSON = os.path.join(HERE, "json", "human_model_comparison.json")
VOCAB = "synthia_filtered"
K = 10
MODELS = ["molmo2-4b", "phi3.5-vision", "gemma4-e4b", "qwen3.5-4b", "gpt-5.4"]
NAMES = {"molmo2-4b": "Molmo2-4B", "phi3.5-vision": "Phi-3.5-V", "gemma4-e4b": "Gemma-4-E4B",
         "qwen3.5-4b": "Qwen3.5-4B", "gpt-5.4": "GPT-5.4"}

# Reference palette (dataviz skill, light mode): fixed categorical order, one slot per model, never cycled.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"molmo2-4b": "#2a78d6", "phi3.5-vision": "#eb6834", "gemma4-e4b": "#1baf7a",
          "qwen3.5-4b": "#eda100", "gpt-5.4": "#e87ba4"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def load_humans():
    """(object, region_id) -> {participant: set(picks)} for part questions."""
    parts = collections.defaultdict(dict)
    in_bank = total = 0
    with open(os.path.join(HERE, "responses_long.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["question"] != "part" or not r["selected"]:
                continue
            picks = {norm(x) for x in r["selected"].split("|") if x.strip()}
            shown = {norm(x) for x in r["options_shown"].split("|") if x.strip()}
            parts[(r["object"], r["region_id"])][r["participant"]] = picks
            total += len(picks)
            in_bank += len(picks & shown)
    return parts, in_bank, total


PROBS = collections.defaultdict(dict)  # model -> (object, region_id) -> {option: prob} (log-prob top-10 only)


def load_models(objects):
    """model -> mode -> (object, region_id) -> ranked list"""
    out = {m: {"generation": {}, "logprob": {}} for m in MODELS}
    for m in MODELS:
        d = os.path.join(HERE, "json", f"affordance_justification_{m}_synthia")
        for obj in objects:
            p = os.path.join(d, f"{obj}_affordance_justification.json")
            if not os.path.isfile(p):
                continue
            for reg in json.load(open(p))["regions"]:
                v = reg["vocabularies"][VOCAB]
                key = (obj, reg["region_id"])
                out[m]["generation"][key] = [norm(x) for x in v.get("generation", {}).get("ranked", [])]
                if "logprob" in v:
                    out[m]["logprob"][key] = [norm(x["option"]) for x in v["logprob"]["ranked"]]
                    PROBS[m][key] = {norm(x["option"]): x["prob"] for x in v["logprob"]["ranked"]}
    return out


def crowd_ranking(others: list) -> list:
    c = collections.Counter(x for s in others for x in s)
    return sorted(c, key=lambda o: (-c[o], o))


def score(ranked: list, picks: set) -> dict:
    hit = [bool(set(ranked[:k]) & picks) for k in range(1, K + 1)]
    top3 = set(ranked[:3])
    jac = len(top3 & picks) / len(top3 | picks) if (top3 | picks) else 0.0
    return {"hit": hit, "jaccard3": jac}


def rankdata(a):
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho; nan when either side is constant."""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_shown():
    """(object, region_id) -> the options shown for that part (identical for every participant)."""
    shown = {}
    with open(os.path.join(HERE, "responses_long.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["question"] == "part":
                shown[(r["object"], r["region_id"])] = [norm(x) for x in r["options_shown"].split("|") if x.strip()]
    return shown


def model_scores(ranked, options, probs=None):
    """Score each shown option: log-prob probability if given, else rank points (10 for #1 ... 1 for #10,
    0 if not in the list)."""
    if probs is not None:
        return [probs.get(o, 0.0) for o in options]
    pts = {o: K - i for i, o in enumerate(ranked[:K])}
    return [pts.get(o, 0) for o in options]


def correlation_analysis(eval_parts, models_probs, models):
    """NVRD-style direct alignment: per part, Spearman rho between the share of people who picked each shown
    option and the model's score for it; averaged over parts, plus one pooled rho over all part x option cells.
    Human reliability for context: each person's 0/1 picks vs the OTHER people's shares (leave-one-out)."""
    shown = load_shown()
    out = {}
    hum_cells = {}
    for key, ppl in eval_parts.items():
        opts = shown[key]
        hum_cells[key] = [sum(o in s for s in ppl.values()) / len(ppl) for o in opts]
    rel = []
    for key, ppl in eval_parts.items():
        opts = shown[key]
        for p, picks in ppl.items():
            others = [s for q, s in ppl.items() if q != p]
            rel.append(spearman([float(o in picks) for o in opts],
                                [sum(o in s for s in others) / len(others) for o in opts]))
    out["human (leave-one-out)"] = {"mean_rho": round(float(np.nanmean(rel)), 4), "pooled_rho": None,
                                    "n_parts": int(np.sum(~np.isnan(rel)))}
    for m in MODELS:
        for mode in ("generation", "logprob"):
            table = models[m][mode]
            if not table:
                continue
            per, xs, ys = [], [], []
            for key in eval_parts:
                if key not in table:
                    continue
                opts = shown[key]
                probs = models_probs[m].get(key) if mode == "logprob" else None
                ms = model_scores(table[key], opts, probs)
                per.append(spearman(hum_cells[key], ms))
                xs += hum_cells[key]
                ys += ms
            out[f"{m}|{mode}"] = {"mean_rho": round(float(np.nanmean(per)), 4),
                                  "pooled_rho": round(spearman(xs, ys), 4),
                                  "n_parts": int(np.sum(~np.isnan(per)))}
    return out


def main():
    humans, in_bank, total_picks = load_humans()
    eval_parts = {k: v for k, v in humans.items() if len(v) >= 2}
    objects = sorted({o for o, _ in humans})
    models = load_models(objects)

    # trials: (part, held-out participant)
    trials = [(key, p) for key, ppl in eval_parts.items() for p in ppl]
    systems = {"human (leave-one-out)": None}
    for m in MODELS:
        systems[f"{m}|generation"] = models[m]["generation"]
        if models[m]["logprob"]:
            systems[f"{m}|logprob"] = models[m]["logprob"]

    results, per_object = {}, collections.defaultdict(dict)
    for name, table in systems.items():
        hits, jacs, obj_hits = [], [], collections.defaultdict(list)
        for key, p in trials:
            picks = eval_parts[key][p]
            if table is None:
                ranked = crowd_ranking([s for q, s in eval_parts[key].items() if q != p])
            else:
                ranked = table.get(key)
                if ranked is None:
                    continue
            s = score(ranked, picks)
            hits.append(s["hit"])
            jacs.append(s["jaccard3"])
            obj_hits[key[0]].append(s["hit"][2])
        h = np.array(hits, dtype=float)
        results[name] = {"n_trials": len(hits), "hit_at_k": h.mean(0).round(4).tolist(),
                         "jaccard_top3": round(float(np.mean(jacs)), 4)}
        for o, v in obj_hits.items():
            per_object[o][name] = round(float(np.mean(v)), 4)

    # label frequencies (all human part answers, not only evaluable parts)
    human_labels = collections.Counter(x for ppl in humans.values() for s in ppl.values() for x in s)
    corr = correlation_analysis(eval_parts, PROBS, models)
    summary = {
        "n_participants": len({p for ppl in humans.values() for p in ppl}),
        "n_parts_answered": len(humans), "n_parts_with_2plus_humans": len(eval_parts),
        "n_trials": len(trials), "share_of_picks_from_bank": round(in_bank / max(total_picks, 1), 4),
        "results": results, "per_object_hit_at_3": per_object, "spearman": corr,
        "top_human_labels": human_labels.most_common(20),
        "caveats": ["bank options came from Molmo2/Gemma-4/Qwen3.5/GPT-5.4 generation top-10s (inflates their agreement)",
                    "Phi-3.5 generation lists were not re-run with shuffled option order",
                    "log-prob scores exist only for each model's top-10; other shown options score 0 (tied)"],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(summary, open(OUT_JSON, "w"), indent=2)

    # ---------------- figures ----------------
    os.makedirs(OUT_FIG, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ceiling = results["human (leave-one-out)"]
    ks = np.arange(1, K + 1)

    # 1) hit@k curves, one panel per mode
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mode in zip(axes, ["generation", "logprob"]):
        ax.plot(ks, ceiling["hit_at_k"], color=INK, lw=2, ls="--", label="Humans (leave-one-out)")
        for m in MODELS:
            r = results.get(f"{m}|{mode}")
            if r:
                ax.plot(ks, r["hit_at_k"], color=COLORS[m], lw=2, marker="o", ms=4, label=NAMES[m])
        ax.set_title(f"{'Generated' if mode == 'generation' else 'Log-prob'} top-k", loc="left", color=INK)
        ax.set_xlabel("k (size of the list checked)")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("P(held-out human's pick is in the top-k)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle(f"How often a model's top-k contains what a person picked  "
                 f"({len(eval_parts)} parts, {len(trials)} person-part trials)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT_FIG, "hit_at_k.png"), dpi=200)
    plt.close(fig)

    # 2) top-1 and top-3 Jaccard, grouped by model (solid = generated, hatched = log-prob),
    #    with the human leave-one-out ceiling as a dashed reference line
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (label, getter) in zip(axes, [
            ("Top-1 matches a person's pick", lambda r: r["hit_at_k"][0]),
            ("Top-3 Jaccard with a person's picks", lambda r: r["jaccard_top3"])]):
        x = np.arange(len(MODELS))
        w = 0.38
        top = getter(ceiling)
        for i, m in enumerate(MODELS):
            for off, mode in ((-w / 2, "generation"), (w / 2, "logprob")):
                r = results.get(f"{m}|{mode}")
                if not r:
                    continue
                v = getter(r)
                ax.bar(i + off, v, width=w, color=COLORS[m], edgecolor=SURFACE, linewidth=2,
                       hatch="///" if mode == "logprob" else None)
                ax.text(i + off, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
                top = max(top, v)
        ax.axhline(getter(ceiling), color=INK, ls="--", lw=1.5)
        ax.text(-0.45, getter(ceiling), f"humans {getter(ceiling):.2f}", va="bottom", ha="left", color=INK, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[m] for m in MODELS], fontsize=9)
        ax.set_title(label, loc="left", color=INK)
        ax.set_ylim(0, top * 1.25)
        ax.grid(axis="y", color=GRID, lw=0.8)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=INK2), Patch(facecolor=INK2, hatch="///", edgecolor=SURFACE)],
               ["generated top-k", "log-prob top-k"], frameon=False, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Model vs human agreement (dashed = how well other people predict a held-out person)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT_FIG, "top1_jaccard.png"), dpi=200)
    plt.close(fig)

    # 3) per-object hit@3 heatmap (generation mode + human ceiling), objects sorted by the human ceiling
    cols_h = ["human (leave-one-out)"] + [f"{m}|generation" for m in MODELS]
    objs = sorted(per_object, key=lambda o: -per_object[o].get("human (leave-one-out)", 0))
    mat = np.array([[per_object[o].get(c, np.nan) for c in cols_h] for o in objs])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(objs) + 1.6))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols_h)))
    ax.set_xticklabels(["Humans"] + [NAMES[m] for m in MODELS], rotation=30, ha="right")
    ax.set_yticks(range(len(objs)))
    ax.set_yticklabels(objs, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if mat[i, j] > 0.55 else INK)
    ax.set_title("Per object: P(person's pick in top-3), generated lists", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "per_object_hit3.png"), dpi=200)
    plt.close(fig)

    # 4) what people pick vs what models put first
    top = [l for l, _ in human_labels.most_common(15)]
    n_human = sum(human_labels.values())
    model_top1 = collections.Counter()
    n_model = 0
    for m in MODELS:
        for key in humans:
            r = models[m]["generation"].get(key)
            if r:
                model_top1[r[0]] += 1
                n_model += 1
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(y - 0.2, [human_labels[l] / n_human for l in top], height=0.38, color=INK2, label="People's picks")
    ax.barh(y + 0.2, [model_top1[l] / max(n_model, 1) for l in top], height=0.38, color="#2a78d6",
            label="Models' top-1 (all five, generated)")
    ax.set_yticks(y)
    ax.set_yticklabels(top)
    ax.invert_yaxis()
    ax.set_xlabel("Share of answers")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("The 15 functionalities people picked most, and how often models rank them first",
                 loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "label_frequencies.png"), dpi=200)
    plt.close(fig)

    # 5) NVRD-style Spearman alignment, grouped by model (solid = generated, hatched = log-prob)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (key_, label) in zip(axes, [("mean_rho", "Mean per-part Spearman \u03c1"),
                                         ("pooled_rho", "Pooled Spearman \u03c1 (all part \u00d7 option cells)")]):
        w = 0.38
        top = 0.05
        for i, m in enumerate(MODELS):
            for off, mode in ((-w / 2, "generation"), (w / 2, "logprob")):
                r = corr.get(f"{m}|{mode}")
                if not r or r[key_] is None:
                    continue
                v = r[key_]
                ax.bar(i + off, v, width=w, color=COLORS[m], edgecolor=SURFACE, linewidth=2,
                       hatch="///" if mode == "logprob" else None)
                ax.text(i + off, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
                top = max(top, v)
        if key_ == "mean_rho":
            hv = corr["human (leave-one-out)"]["mean_rho"]
            ax.axhline(hv, color=INK, ls="--", lw=1.5)
            ax.text(-0.45, hv, f"person vs others {hv:.2f}", va="bottom", ha="left", color=INK, fontsize=9)
            top = max(top, hv)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([NAMES[m] for m in MODELS], fontsize=9)
        ax.set_title(label, loc="left", color=INK)
        ax.set_ylim(min(0, ax.get_ylim()[0]), top * 1.25)
        ax.grid(axis="y", color=GRID, lw=0.8)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=INK2), Patch(facecolor=INK2, hatch="///", edgecolor=SURFACE)],
               ["generated top-k (rank points)", "log-prob (probability)"], frameon=False, loc="lower center",
               ncol=2, fontsize=9)
    fig.suptitle("Direct alignment: share of people picking each shown option vs the model's score for it",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT_FIG, "spearman_alignment.png"), dpi=200)
    plt.close(fig)

    # ---------------- printed summary ----------------
    print(f"{summary['n_participants']} participants; {summary['n_parts_answered']} parts answered, "
          f"{summary['n_parts_with_2plus_humans']} with >= 2 people ({summary['n_trials']} trials)")
    print(f"{summary['share_of_picks_from_bank']:.0%} of picks came from the option bank (the rest typed)")
    print(f"{'system':32s} {'hit@1':>6} {'hit@3':>6} {'hit@10':>6} {'jac@3':>6}")
    for n, r in results.items():
        print(f"{n:32s} {r['hit_at_k'][0]:6.2f} {r['hit_at_k'][2]:6.2f} {r['hit_at_k'][9]:6.2f} {r['jaccard_top3']:6.2f}")
    print(f"\n{'Spearman (NVRD-style)':32s} {'mean rho':>9} {'pooled':>7} {'parts':>6}")
    for n, r in corr.items():
        pooled = "" if r["pooled_rho"] is None else f"{r['pooled_rho']:7.2f}"
        print(f"{n:32s} {r['mean_rho']:9.2f} {pooled:>7} {r['n_parts']:6d}")
    print("caveats:", *summary["caveats"], sep="\n  - ")
    print(f"figures -> {OUT_FIG}\nnumbers -> {OUT_JSON}")


if __name__ == "__main__":
    main()
