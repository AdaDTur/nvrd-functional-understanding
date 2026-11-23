"""
NVRD-style direct alignment between human and model functionality judgments (Spearman rho), at three
levels, each against a split-half human ceiling.

  part     per part: share of people who picked each shown option vs the model's score for it
           (generated list: 10 points for #1 ... 1 for #10, 0 if absent; log-prob: probability, 0 outside
           its top-10). Averaged over parts, and per object.
  object   the whole-object question: share of people picking each shown overall option vs the model's
           object score (generated: summed rank points over the object's parts; log-prob: mean probability).
  grouped  part level again, but over the coarse functional groups in json/synthia_function_groups.json
           (people's picks and typed answers mapped to groups; a model's group score = its best score
           for any label in that group).

Human ceiling (split-half, the NVRD-equivalent of model-vs-humans): for parts/objects with >= 4 people,
split them at random into two halves (200 splits), correlate the halves' pick shares, and average.
Reported raw and Spearman-Brown corrected (2r / (1 + r), the reliability of the full group).

Model "participants" (json/model_participant_<model>.json, from code/model_probing/model_as_participant.py)
pick up to 3 options like a person, so they are scored like one: Spearman between its 0/1 picks and the
people's shares, next to each person's 0/1 picks vs the OTHER people's shares.

Outputs: figures/human_vs_model/alignment_*.png and json/human_model_alignment.json
Usage (from nvrd/affordance-test/): ../venv_vlm/bin/python3 code/plotting/human_model_alignment.py
"""
import collections
import csv
import glob
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from compare_human_model import (COLORS, GRID, HERE, INK, INK2, K, MODELS, NAMES, PROBS, SURFACE, load_models,  # noqa: E402
                                 norm, spearman)

OUT_FIG = os.path.join(HERE, "figures", "human_vs_model")
OUT_JSON = os.path.join(HERE, "json", "human_model_alignment.json")
MIN_PEOPLE = 2      # to correlate a model with the people's shares
MIN_SPLIT = 4       # to split people into two halves
N_SPLITS = 200
DIVERGING = ["#eb6834", "#f3a37f", "#f0efec", "#86b6ef", "#2a78d6"]  # orange (-) .. neutral .. blue (+)


def load_people():
    parts = collections.defaultdict(dict)
    overall = collections.defaultdict(dict)
    shown, shown_overall = {}, {}
    with open(os.path.join(HERE, "responses_long.csv"), newline="") as f:
        for r in csv.DictReader(f):
            picks = {norm(x) for x in r["selected"].split("|") if x.strip()}
            opts = [norm(x) for x in r["options_shown"].split("|") if x.strip()]
            if r["question"] == "part":
                parts[(r["object"], r["region_id"])][r["participant"]] = picks
                shown[(r["object"], r["region_id"])] = opts
            else:
                overall[r["object"]][r["participant"]] = picks
                shown_overall[r["object"]] = opts
    return parts, overall, shown, shown_overall


def shares(people: dict, items: list, to_items=lambda picks: picks) -> list:
    sets = [to_items(p) for p in people.values()]
    return [sum(i in s for s in sets) / len(sets) for i in items]


def split_half(people: dict, items: list, to_items=lambda picks: picks, seed=0):
    ids = sorted(people)
    if len(ids) < MIN_SPLIT:
        return None
    rng = random.Random(seed)
    rs = []
    for _ in range(N_SPLITS):
        rng.shuffle(ids)
        a, b = ids[:len(ids) // 2], ids[len(ids) // 2:]
        rs.append(spearman(shares({i: people[i] for i in a}, items, to_items),
                           shares({i: people[i] for i in b}, items, to_items)))
    r = float(np.nanmean(rs)) if not all(np.isnan(rs)) else float("nan")
    return r


def spearman_brown(r):
    return 2 * r / (1 + r) if r is not None and not np.isnan(r) and r > -1 else float("nan")


def gen_points(ranked):
    return {o: K - i for i, o in enumerate(ranked[:K])}


def load_groups():
    d = json.load(open(os.path.join(HERE, "json", "synthia_function_groups.json")))
    label2group = {norm(l): g for g, ls in d["groups"].items() for l in ls}
    typed = {norm(k): v for k, v in d.get("typed", {}).items()}

    def to_groups(picks):
        out = set()
        for p in picks:
            if p in label2group:
                out.add(label2group[p])
            else:
                out.update(typed.get(p, []))
        return out
    return label2group, to_groups


def main():
    parts, overall, shown, shown_overall = load_people()
    objects = sorted({o for o, _ in parts})
    models = load_models(objects)
    label2group, to_groups = load_groups()
    systems = [(m, mode) for m in MODELS for mode in ("generation", "logprob") if models[m][mode]]

    def model_scores(m, mode, key):
        if mode == "logprob":
            return PROBS[m].get(key)
        r = models[m]["generation"].get(key)
        return gen_points(r) if r is not None else None

    res = {"part": {}, "object": {}, "grouped": {}, "per_object": collections.defaultdict(dict),
           "participants": {}}
    eval_parts = {k: v for k, v in parts.items() if len(v) >= MIN_PEOPLE}

    # ---- part level ----
    ceil = [split_half(ppl, shown[key], seed=i) for i, (key, ppl) in enumerate(eval_parts.items())]
    ceil_by_obj = collections.defaultdict(list)
    for (key, _), c in zip(eval_parts.items(), ceil):
        if c is not None:
            ceil_by_obj[key[0]].append(c)
    ceil = [c for c in ceil if c is not None]
    res["part"]["humans split-half"] = {"rho": float(np.nanmean(ceil)), "n": len(ceil)}
    for o, v in ceil_by_obj.items():
        res["per_object"][o]["humans split-half"] = float(np.nanmean(v))
    for m, mode in systems:
        per, by_obj = [], collections.defaultdict(list)
        for key, ppl in eval_parts.items():
            sc = model_scores(m, mode, key)
            if sc is None:
                continue
            r = spearman(shares(ppl, shown[key]), [sc.get(o, 0) for o in shown[key]])
            per.append(r)
            by_obj[key[0]].append(r)
        res["part"][f"{m}|{mode}"] = {"rho": float(np.nanmean(per)), "n": int(np.sum(~np.isnan(per)))}
        for o, v in by_obj.items():
            res["per_object"][o][f"{m}|{mode}"] = float(np.nanmean(v)) if not all(np.isnan(v)) else float("nan")

    # ---- whole-object level ----
    eval_objs = {o: v for o, v in overall.items() if len(v) >= MIN_PEOPLE}
    oc = [split_half(ppl, shown_overall[o], seed=i) for i, (o, ppl) in enumerate(eval_objs.items())]
    oc = [c for c in oc if c is not None]
    res["object"]["humans split-half"] = {"rho": float(np.nanmean(oc)) if oc else float("nan"), "n": len(oc)}
    for m, mode in systems:
        per = []
        for o, ppl in eval_objs.items():
            keys = [k for k in parts if k[0] == o] + [k for k in models[m][mode] if k[0] == o and k not in parts]
            keys = sorted(set(keys))
            agg = collections.defaultdict(float)
            n = 0
            for key in keys:
                sc = model_scores(m, mode, key)
                if sc is None:
                    continue
                n += 1
                for lab, v in sc.items():
                    agg[lab] += v
            if not n:
                continue
            if mode == "logprob":
                agg = {k: v / n for k, v in agg.items()}
            per.append(spearman(shares(ppl, shown_overall[o]), [agg.get(x, 0) for x in shown_overall[o]]))
        res["object"][f"{m}|{mode}"] = {"rho": float(np.nanmean(per)), "n": int(np.sum(~np.isnan(per)))}

    # ---- grouped (coarse functional groups), part level ----
    def items_for(key, ppl):
        g = {label2group[o] for o in shown[key] if o in label2group}
        for picks in ppl.values():
            g |= to_groups(picks)
        return sorted(g)
    gc = [split_half(ppl, items_for(key, ppl), to_groups, seed=i) for i, (key, ppl) in enumerate(eval_parts.items())]
    gc = [c for c in gc if c is not None]
    res["grouped"]["humans split-half"] = {"rho": float(np.nanmean(gc)), "n": len(gc)}
    for m, mode in systems:
        per = []
        for key, ppl in eval_parts.items():
            sc = model_scores(m, mode, key)
            if sc is None:
                continue
            items = items_for(key, ppl)
            gscore = collections.defaultdict(float)
            for lab, v in sc.items():
                g = label2group.get(lab)
                if g:
                    gscore[g] = max(gscore[g], v)
            per.append(spearman(shares(ppl, items, to_groups), [gscore.get(g, 0) for g in items]))
        res["grouped"][f"{m}|{mode}"] = {"rho": float(np.nanmean(per)), "n": int(np.sum(~np.isnan(per)))}

    # ---- overlap on labels vs functional groups (simple, tie-free complement to Spearman) ----
    # people: every pair of people on the same part; models: the model's top-3 vs each person;
    # model participants: their picks vs each person. "any" = share >= 1 pick; "jaccard" = |A&B| / |A|B|.
    import itertools

    def overlap(a, b):
        return (bool(a & b), len(a & b) / len(a | b) if (a | b) else 0.0)
    ov = {}
    for lvl, f in (("labels", lambda x: x), ("groups", to_groups)):
        pairs = [overlap(f(a), f(b)) for ppl in eval_parts.values() for a, b in itertools.combinations(ppl.values(), 2)]
        ov.setdefault("two people", {})[lvl] = {"any": float(np.mean([p[0] for p in pairs])),
                                                "jaccard": float(np.mean([p[1] for p in pairs])), "n": len(pairs)}
        for m, mode in systems:
            vals = []
            for key, ppl in eval_parts.items():
                sc = model_scores(m, mode, key)
                if sc is None:
                    continue
                top3 = set(sorted(sc, key=lambda o: -sc[o])[:3])
                vals += [overlap(f(top3), f(pk)) for pk in ppl.values()]
            ov.setdefault(f"{m}|{mode}", {})[lvl] = {"any": float(np.mean([v[0] for v in vals])),
                                                    "jaccard": float(np.mean([v[1] for v in vals])), "n": len(vals)}
        for path in sorted(glob.glob(os.path.join(HERE, "json", "model_participant_*.json"))):
            d = json.load(open(path))
            vals = []
            for a in d["parts"]:
                key = (a["object"], a["region_id"])
                if key in eval_parts:
                    mine = {norm(x) for x in a["selected"]}
                    vals += [overlap(f(mine), f(pk)) for pk in eval_parts[key].values()]
            if vals:
                ov.setdefault(f"participant {d['model']}", {})[lvl] = {
                    "any": float(np.mean([v[0] for v in vals])), "jaccard": float(np.mean([v[1] for v in vals])),
                    "n": len(vals)}
    res["overlap"] = ov

    # ---- model participants (pick up to 3 like a person) ----
    loo = []
    for key, ppl in eval_parts.items():
        for p, picks in ppl.items():
            others = {q: s for q, s in ppl.items() if q != p}
            loo.append(spearman([float(o in picks) for o in shown[key]], shares(others, shown[key])))
    res["participants"]["a person vs the other people"] = {"rho": float(np.nanmean(loo)), "n": len(loo)}
    for path in sorted(glob.glob(os.path.join(HERE, "json", "model_participant_*.json"))):
        d = json.load(open(path))
        per = []
        for a in d["parts"]:
            key = (a["object"], a["region_id"])
            if key in eval_parts:
                picks = {norm(x) for x in a["selected"]}
                per.append(spearman([float(o in picks) for o in shown[key]], shares(eval_parts[key], shown[key])))
        objs = []
        for a in d.get("objects", []):
            if a["object"] in eval_objs:
                picks = {norm(x) for x in a["selected"]}
                o = a["object"]
                objs.append(spearman([float(x in picks) for x in shown_overall[o]], shares(eval_objs[o], shown_overall[o])))
        res["participants"][d["model"]] = {"rho": float(np.nanmean(per)) if per else float("nan"), "n": len(per),
                                           "object_rho": float(np.nanmean(objs)) if objs else float("nan")}

    # ---- like-for-like: every system judged against the SAME reference (the other people) ----
    # For each part and each held-out person p, the reference is the OTHER people's pick shares on that part.
    # Scored against it: p's own picks; each ranked model's top-n where n = the number of options p picked
    # (same pick count as the person); each model participant's own picks. Metrics: Spearman of the 0/1 picks
    # vs the others' shares over the shown options, and mean Jaccard with each other person's picks.
    # Parts need >= 2 people (so the reference has >= 1 person); every system uses exactly the same trials.
    participant_files = {}
    for path in sorted(glob.glob(os.path.join(HERE, "json", "model_participant_*.json"))):
        d = json.load(open(path))
        participant_files[d["model"]] = {(a["object"], a["region_id"]): {norm(x) for x in a["selected"]}
                                         for a in d["parts"]}
    fair = collections.defaultdict(lambda: {"rho": [], "jaccard": [], "n_picks": []})
    for key, ppl in eval_parts.items():
        opts = shown[key]
        for p, picks in ppl.items():
            others = {q: s_ for q, s_ in ppl.items() if q != p}
            ref = shares(others, opts)
            n = max(1, min(len(picks & set(opts)) or len(picks), K))

            def add(name, mine):
                fair[name]["rho"].append(spearman([float(o in mine) for o in opts], ref))
                fair[name]["jaccard"].append(float(np.mean([len(mine & o_) / len(mine | o_) if (mine | o_) else 0
                                                            for o_ in others.values()])))
                fair[name]["n_picks"].append(len(mine))
            add("a person", picks)
            for m, mode in systems:
                sc = model_scores(m, mode, key)
                if sc is not None:
                    ranked = sorted(sc, key=lambda o: -sc[o])
                    add(f"{m}|{mode} (top-n)", set(ranked[:n]))
            for name, table in participant_files.items():
                if key in table:
                    add(f"participant {name}", table[key])
    res["like_for_like"] = {k: {"rho": float(np.nanmean(v["rho"])), "jaccard": float(np.mean(v["jaccard"])),
                                "mean_picks": float(np.mean(v["n_picks"])), "n": len(v["rho"])}
                            for k, v in fair.items()}

    for lvl in ("part", "object", "grouped"):
        c = res[lvl]["humans split-half"]
        c["rho_spearman_brown"] = spearman_brown(c["rho"])
    res["per_object"] = dict(res["per_object"])
    res["notes"] = {"min_people": MIN_PEOPLE, "min_people_for_split_half": MIN_SPLIT, "n_splits": N_SPLITS,
                    "n_parts": len(eval_parts), "n_objects_overall": len(eval_objs)}
    json.dump(res, open(OUT_JSON, "w"), indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    # ---------------- figures ----------------
    os.makedirs(OUT_FIG, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1) three levels side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (lvl, title) in zip(axes, [("part", "Part level"), ("grouped", "Part level, functional groups"),
                                       ("object", "Whole object")]):
        w = 0.38
        for i, m in enumerate(MODELS):
            for off, mode in ((-w / 2, "generation"), (w / 2, "logprob")):
                r = res[lvl].get(f"{m}|{mode}")
                if not r or np.isnan(r["rho"]):
                    continue
                v = r["rho"]
                ax.bar(i + off, v, width=w, color=COLORS[m], edgecolor=SURFACE, linewidth=2,
                       hatch="///" if mode == "logprob" else None)
                ax.text(i + off, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
        c = res[lvl]["humans split-half"]
        if not np.isnan(c["rho"]):
            ax.axhline(c["rho"], color=INK, ls="--", lw=1.5)
            ax.text(-0.45, c["rho"], f"half vs half of people {c['rho']:.2f} (n={c['n']})", va="bottom",
                    ha="left", color=INK, fontsize=8.5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([NAMES[m] for m in MODELS], fontsize=8.5, rotation=20)
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("Spearman ρ with people's pick shares")
    fig.legend([Patch(facecolor=INK2), Patch(facecolor=INK2, hatch="///", edgecolor=SURFACE)],
               ["generated top-k (rank points)", "log-prob (probability)"], frameon=False, loc="lower center",
               ncol=2, fontsize=9)
    fig.suptitle("Human-model alignment (Spearman ρ); dashed = agreement between two random halves of people",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT_FIG, "alignment_levels.png"), dpi=200)
    plt.close(fig)

    # 2) per object heatmap (part-level rho)
    cols = ["humans split-half"] + [f"{m}|{mode}" for m, mode in systems]
    labels = ["People\n(half vs half)"] + [f"{NAMES[m]}\n{'gen' if mode == 'generation' else 'log-prob'}"
                                          for m, mode in systems]
    objs = sorted(res["per_object"], key=lambda o: -np.nan_to_num(np.nanmean(
        [res["per_object"][o].get(c, np.nan) for c in cols[1:]]), nan=-9))
    mat = np.array([[res["per_object"][o].get(c, np.nan) for c in cols] for o in objs], dtype=float)
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(12, 0.34 * len(objs) + 2))
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(objs)))
    ax.set_yticklabels(objs, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_title("Per object: mean part-level Spearman ρ with people (blank = not enough people / data)",
                 loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "alignment_per_object.png"), dpi=200)
    plt.close(fig)

    # 3) like-for-like bars
    lf = res["like_for_like"]
    order = ["a person"] + [k for k in lf if k.startswith("participant")] + \
            [f"{m}|{mode} (top-n)" for m, mode in systems if f"{m}|{mode} (top-n)" in lf]

    def color_of(k):
        if k == "a person":
            return INK2
        if k.startswith("participant"):
            return "#6250d6"  # violet: categorical slot 7, not used by the five models
        return COLORS[k.split("|")[0]]

    def label_of(k):
        if k == "a person":
            return "A person"
        if k.startswith("participant"):
            return k.replace("participant google/gemini-3.8-flash", "Gemini 3.8 Flash\nparticipant").replace(" pick1", ", 1 pick")
        m, rest = k.split("|")
        return f"{NAMES[m]}\n{'gen' if rest.startswith('generation') else 'log-prob'} top-n"
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (metric, title) in zip(axes, [("rho", "Spearman \u03c1 with the other people's pick shares"),
                                          ("jaccard", "Mean Jaccard with each other person's picks")]):
        vals = [lf[k][metric] for k in order]
        x = np.arange(len(order))
        bars = ax.bar(x, vals, color=[color_of(k) for k in order], edgecolor=SURFACE, linewidth=2, width=0.75)
        for b, k in zip(bars, order):
            if "logprob" in k:
                b.set_hatch("///")
        ax.axhline(lf["a person"][metric], color=INK, ls="--", lw=1.2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([label_of(k) for k in order], fontsize=7, rotation=45, ha="right")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
    fig.suptitle("Like-for-like: each held-out person and each model judged against the same other people, "
                 "with the same number of picks (dashed = a person)", x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "alignment_like_for_like.png"), dpi=200)
    plt.close(fig)

    # 4) how typical each responder's answers are: how many of the OTHER
    #    people gave the same answer. Each person contributes every pick; each model one answer per trial
    #    (its single pick / top-1), on the same (part, held-out person) trials and the same reference group.
    pop = collections.defaultdict(list)
    for key, ppl in eval_parts.items():
        opts = set(shown[key])
        for p, picks in ppl.items():
            others = [s_ for q, s_ in ppl.items() if q != p]

            def share(o):
                return sum(o in s_ for s_ in others) / len(others)
            pop["A person"] += [share(o) for o in picks if o in opts]
            for name, table in participant_files.items():
                if name.endswith("pick1") and table.get(key):
                    pop["Gemini 3.8 Flash\n(participant, 1 pick)"] += [share(o) for o in table[key] if o in opts][:1]
            for m, mode, lab in (("gpt-5.4", "generation", "GPT-5.4\n(top-1, generated)"),
                                 ("qwen3.5-4b", "logprob", "Qwen3.5-4B\n(top-1, log-prob)"),
                                 ("gemma4-e4b", "logprob", "Gemma-4-E4B\n(top-1, log-prob)"),
                                 ("molmo2-4b", "logprob", "Molmo2-4B\n(top-1, log-prob)")):
                sc = model_scores(m, mode, key)
                if sc:
                    top = max(sc, key=sc.get)
                    pop[lab].append(share(top))
    rows = list(pop)
    bins = [("nobody else", lambda v: v == 0), ("some (under half)", lambda v: 0 < v < 0.5),
            ("most (half or more)", lambda v: v >= 0.5)]
    seg_cols = ["#cde2fb", "#5598e7", "#184f95"]  # ordinal blue ramp
    fracs = np.array([[np.mean([f(v) for v in pop[r]]) for _, f in bins] for r in rows])
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(rows) + 1.8))
    left = np.zeros(len(rows))
    y = np.arange(len(rows))
    for j, ((name, _), col) in enumerate(zip(bins, seg_cols)):
        ax.barh(y, fracs[:, j], left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.62, label=name)
        for i, v in enumerate(fracs[:, j]):
            if v >= 0.07:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if j > 0 else INK)
        left += fracs[:, j]
    for i, r in enumerate(rows):
        ax.text(1.01, i, f"avg {np.mean(pop[r]):.0%} of others", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of answers")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(title="How many of the OTHER people gave the same answer", frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9, title_fontsize=9)
    ax.set_title("How common is each answer among the other people on that part?",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "answer_typicality.png"), dpi=200)
    plt.close(fig)
    res["answer_typicality"] = {r.replace("\n", " "): {"mean_share_of_others": float(np.mean(pop[r])),
                                                        "n": len(pop[r]),
                                                        **{b: float(fracs[i, j]) for j, (b, _) in enumerate(bins)}}
                                for i, r in enumerate(rows)}
    json.dump(res, open(OUT_JSON, "w"), indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    # ---------------- printed summary ----------------
    n = res["notes"]
    print(f"{n['n_parts']} parts and {n['n_objects_overall']} whole objects with >= {MIN_PEOPLE} people")
    print(f"{'':30s} {'part':>7} {'grouped':>8} {'object':>7}")
    for name in ["humans split-half"] + [f"{m}|{mode}" for m, mode in systems]:
        row = [res[l].get(name, {}).get("rho", float("nan")) for l in ("part", "grouped", "object")]
        print(f"{name:30s} " + " ".join(f"{v:7.2f}" if not np.isnan(v) else "      -" for v in row))
    print(f"{'  (Spearman-Brown corrected)':30s} " + " ".join(
        f"{res[l]['humans split-half']['rho_spearman_brown']:7.2f}" for l in ("part", "grouped", "object")))
    print("split-half n (parts/objects with >= 4 people):",
          {l: res[l]["humans split-half"]["n"] for l in ("part", "grouped", "object")})
    print(f"\noverlap with a person's picks (share >= 1 / Jaccard)      labels         groups")
    for name, v in res["overlap"].items():
        print(f"  {name:40s} {v['labels']['any']:.2f} / {v['labels']['jaccard']:.2f}    "
              f"{v['groups']['any']:.2f} / {v['groups']['jaccard']:.2f}")
    print(f"\nlike-for-like vs the OTHER people ({res['like_for_like']['a person']['n']} person-part trials)")
    print(f"  {'':46s} {'rho':>5} {'jaccard':>8} {'picks':>6}")
    for k, v in sorted(res["like_for_like"].items(), key=lambda kv: -kv[1]["rho"]):
        print(f"  {k:46s} {v['rho']:5.2f} {v['jaccard']:8.2f} {v['mean_picks']:6.2f}")
    if len(res["participants"]) > 1:
        print("\nas a participant (0/1 picks vs people's shares):")
        for k, v in res["participants"].items():
            print(f"  {k:32s} part {v['rho']:.2f}" + (f"  object {v['object_rho']:.2f}" if "object_rho" in v else ""))
    print(f"figures -> {OUT_FIG}/alignment_*.png\nnumbers -> {OUT_JSON}")


if __name__ == "__main__":
    main()
