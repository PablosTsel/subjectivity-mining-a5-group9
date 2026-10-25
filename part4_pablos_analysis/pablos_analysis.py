"""Assignment 5, Group 9, Pablos' part.

1. Exports the fBERT probabilities of the A4 models for both settings.
2. Pearson correlations between the three base models (probabilities, labels and errors).
3. Quantitative comparison of the ensembles with the best single model, with McNemar tests.
4. Quantitative error analysis of the best ensemble (stacking) in both settings.
5. Summary of the manual error categories of the cross-domain stacking errors.

All comparisons use Martin's matched model pool, so every ensemble is built from the same
three base models. Run from this folder: python pablos_analysis.py
"""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MARTIN = ROOT / "part3_martin_stacking/stacking_results"
A4_FBERT = HERE / "a4_fbert_results"  # fBERT outputs of Assignment 4
OUT = HERE / "results"
PICS = HERE / "pics"
SETTINGS = {"in_domain": "In-domain", "cross_domain": "Cross-domain"}
BASE = ["bert", "hatebert", "fbert"]
NAMES = {"bert": "BERT", "hatebert": "HateBERT", "fbert": "fBERT"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
# One colour per system, the same in every figure.
COLORS = {"BERT": "#2a78d6", "HateBERT": "#eb6834", "fBERT": "#1baf7a",
          "Hard voting": "#eda100", "Soft voting": "#e87ba4", "Stacking": "#4a3aa7"}
# Same short swear word list as in Assignment 4.
EXPLICIT = re.compile(
    r"\b(?:fuck\w*|shit\w*|bitch\w*|ass|asshole\w*|damn\w*|crap\w*|dick\w*|pussy|cunt\w*|"
    r"bastard\w*|nigg\w*|whore\w*|slut\w*|idiot\w*|stupid\w*|moron\w*|dumb\w*|retard\w*|"
    r"scum\w*|hell|piss\w*|motherf\w*|fag\w*|douche\w*|jackass\w*)\b", re.IGNORECASE)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_axisbelow(True)


def export_fbert_probabilities():
    """fBERT probabilities of the A4 models (used by Helian's soft voting)."""
    for s in SETTINGS:
        r = pd.read_csv(A4_FBERT / f"results_fbert_{s}.csv")
        out = r[["id", "text", "labels", "prediction", "probability_offensive"]].copy()
        out["probability_non_offensive"] = 1 - out.probability_offensive
        assert np.array_equal(out.prediction, (r.logit_1 > r.logit_0).astype(int))
        out.to_csv(OUT / f"fbert_probs_{s}.csv", index=False)
        # Helian's soft voting used these probabilities, its fBERT scores match the A4 scores.
        # (fbert_probs_*.csv in Martin's folder are his retrained fBERT from the matched pool.)
        a4 = pd.read_csv(A4_FBERT / f"metrics_fbert_{s}.csv").macro_f1.iloc[0]
        helian = pd.read_csv(ROOT / "part2_helian_soft_voting/metrics_soft_voting_vs_individual.csv")
        assert np.isclose(helian.query("model == 'fbert' and setting == @SETTINGS[@s]").macro_f1.iloc[0], a4)


def load_pool(s):
    """Test probabilities and labels of the matched pool, plus all system predictions."""
    feats = pd.read_csv(MARTIN / f"test_features_{s}.csv")
    preds = {NAMES[m]: pd.read_csv(MARTIN / f"results_{m}_matched_{s}.csv").prediction.values for m in BASE}
    preds["Hard voting"] = pd.read_csv(MARTIN / f"results_hard_voting_matched_{s}.csv").prediction.values
    preds["Soft voting"] = pd.read_csv(MARTIN / f"results_soft_voting_matched_{s}.csv").prediction.values
    preds["Stacking"] = pd.read_csv(MARTIN / f"results_stacking_{s}.csv").prediction.values
    for m in BASE:  # labels of the base models are the argmax of their probabilities
        assert np.array_equal(preds[NAMES[m]], (feats[f"{m}_prob"] > 0.5).astype(int))
    return feats, preds


def correlations(feats, preds, setting):
    y = feats.labels.values
    rows = []
    for i, a in enumerate(BASE):
        for b in BASE[i + 1:]:
            pa, pb = feats[f"{a}_prob"].values, feats[f"{b}_prob"].values
            la, lb = preds[NAMES[a]], preds[NAMES[b]]
            ea, eb = (la != y).astype(int), (lb != y).astype(int)
            rows.append({"setting": setting, "pair": f"{NAMES[a]} / {NAMES[b]}",
                         "r_probabilities": pearsonr(pa, pb)[0], "r_labels": pearsonr(la, lb)[0],
                         "r_errors": pearsonr(ea, eb)[0], "label_agreement": (la == lb).mean()})
    labels = np.vstack([preds[NAMES[m]] for m in BASE])
    correct = labels == y
    summary = {"setting": setting, "all_three_agree": int((labels.min(0) == labels.max(0)).sum()),
               "all_three_correct": int(correct.all(0).sum()), "all_three_wrong": int((~correct).all(0).sum()),
               "at_least_one_correct": int(correct.any(0).sum())}
    return rows, summary


def mcnemar(a_correct, b_correct):
    only_a, only_b = int((a_correct & ~b_correct).sum()), int((~a_correct & b_correct).sum())
    return only_a, only_b, binomtest(only_a, only_a + only_b, 0.5).pvalue


def comparisons(feats, preds, setting):
    y = feats.labels.values
    ok = {k: v == y for k, v in preds.items()}
    rows = []
    for a, b in [("Stacking", "fBERT"), ("Soft voting", "fBERT"), ("Hard voting", "fBERT"), ("Stacking", "Soft voting")]:
        only_a, only_b, p = mcnemar(ok[a], ok[b])
        rows.append({"setting": setting, "system_a": a, "system_b": b, "only_a_correct": only_a,
                     "only_b_correct": only_b, "mcnemar_exact_p": p})
    return rows


def stacking_errors(feats, preds, setting):
    y = feats.labels.values
    p = preds["Stacking"]
    words = feats.text.str.split().str.len()
    bins = pd.cut(words, [-1, 15, 30, np.inf], labels=["0-15 words", "16-30 words", "31+ words"])
    explicit = feats.text.str.contains(EXPLICIT).values
    rows = []
    for g in bins.cat.categories:
        k = (bins == g).values
        rows.append({"setting": setting, "group": g, "n": int(k.sum()), "error_rate": float((p[k] != y[k]).mean())})
    for flag, name in [(True, "offensive, with swear word"), (False, "offensive, without swear word")]:
        k = (explicit == flag) & (y == 1)
        rows.append({"setting": setting, "group": name, "n": int(k.sum()), "error_rate": float((p[k] != 1).mean())})
    return rows


def fig_macro_f1(metrics):
    systems = ["BERT", "HateBERT", "fBERT", "Hard voting", "Soft voting", "Stacking"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x, w = np.arange(2), 0.13
    for i, sname in enumerate(systems):
        vals = [metrics.loc[(sname, st), "macro_f1"] for st in SETTINGS.values()]
        bars = ax.bar(x + (i - 2.5) * (w + 0.008), vals, w, label=sname, color=COLORS[sname])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, color=INK, rotation=90)
    ax.set_xticks(x, list(SETTINGS.values()), color=INK)
    ax.set_ylim(0.6, 0.88)
    ax.set_ylabel("Macro-F1", color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(PICS / "macro_f1_ensembles.png", dpi=200)
    plt.close(fig)


def fig_correlations(corr):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, st in zip(axes, SETTINGS.values()):
        m = np.eye(3)
        sub = corr[corr.setting == st].set_index("pair")["r_probabilities"]
        for i, a in enumerate(BASE):
            for j, b in enumerate(BASE):
                if i < j:
                    m[i, j] = m[j, i] = sub[f"{NAMES[a]} / {NAMES[b]}"]
        ax.imshow(m, cmap="Blues", vmin=0.5, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if v > 0.82 else INK)
        ticks = [NAMES[b] for b in BASE]
        ax.set_xticks(range(3), ticks, fontsize=9)
        ax.set_yticks(range(3), ticks, fontsize=9)
        ax.set_title(st, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(PICS / "pearson_probabilities.png", dpi=200)
    plt.close(fig)


def fig_confusion_stacking(feats_by_setting, preds_by_setting):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, s in zip(axes, SETTINGS):
        cm = confusion_matrix(feats_by_setting[s].labels, preds_by_setting[s]["Stacking"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=620)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=12, color="white" if v > 330 else INK)
        ax.set_xticks([0, 1], ["Pred. non-off.", "Pred. off."], fontsize=8)
        ax.set_yticks([0, 1], ["True non-off.", "True off."], fontsize=8)
        ax.set_title(f"Stacking, {SETTINGS[s]}", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(PICS / "confusion_stacking.png", dpi=200)
    plt.close(fig)


def fig_error_categories(ann):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (etype, title, color) in zip(axes, [("false_positive", "False positives (39)", "#eb6834"),
                                                ("false_negative", "False negatives (89)", "#2a78d6")]):
        counts = ann[ann.error_type == etype].category.value_counts().sort_values()
        bars = ax.barh(counts.index, counts.values, color=color, height=0.6)
        ax.bar_label(bars, padding=3, fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_xlim(0, counts.max() * 1.18)
        style(ax)
    fig.tight_layout()
    fig.savefig(PICS / "error_categories_cross_domain.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    PICS.mkdir(exist_ok=True)
    export_fbert_probabilities()

    metrics = pd.read_csv(MARTIN / "metrics_matched_pool_comparison.csv")
    metrics["model"] = metrics.model.replace({"bert": "BERT", "hatebert": "HateBERT", "fbert": "fBERT",
                                              "Hard voting (matched)": "Hard voting",
                                              "Soft voting (matched)": "Soft voting"})
    metrics = metrics.set_index(["model", "setting"])

    corr_rows, agree_rows, comp_rows, err_rows = [], [], [], []
    feats_all, preds_all = {}, {}
    for s, st in SETTINGS.items():
        feats, preds = load_pool(s)
        feats_all[s], preds_all[s] = feats, preds
        r, a = correlations(feats, preds, st)
        corr_rows += r
        agree_rows.append(a)
        comp_rows += comparisons(feats, preds, st)
        err_rows += stacking_errors(feats, preds, st)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "pearson_correlations.csv", index=False)
    pd.DataFrame(agree_rows).to_csv(OUT / "base_model_agreement.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(OUT / "mcnemar_ensembles_vs_fbert.csv", index=False)
    pd.DataFrame(err_rows).to_csv(OUT / "stacking_error_breakdown.csv", index=False)

    ann = pd.read_csv(HERE / "error_annotations_cross_domain.csv")
    summary = ann.groupby(["error_type", "category"]).size().rename("n").reset_index()
    summary["percent_of_type"] = 100 * summary.n / summary.groupby("error_type").n.transform("sum")
    summary.to_csv(OUT / "error_categories_summary.csv", index=False)

    fig_macro_f1(metrics)
    fig_correlations(corr)
    fig_confusion_stacking(feats_all, preds_all)
    fig_error_categories(ann)

    pd.set_option("display.width", 200)
    for name in ["pearson_correlations", "base_model_agreement", "mcnemar_ensembles_vs_fbert",
                 "stacking_error_breakdown", "error_categories_summary"]:
        print("==", name)
        print(pd.read_csv(OUT / f"{name}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
