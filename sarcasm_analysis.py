"""
Task 2: Sarcasm Headline Analysis
=================================
Goal: find what distinguishes SARCASTIC headlines (from The Onion) from
GENUINE news headlines (from HuffPost). No classifier is built -- this is
purely about identifying patterns.

Dataset: News Headlines Dataset for Sarcasm Detection (rmisra).
Fields: is_sarcastic (1 = Onion, 0 = HuffPost), headline, article_link.

Run:  python3 sarcasm_analysis.py
Outputs: prints text statistics + findings, saves comparison charts as PNGs.
"""

import json
import re
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "Sarcasm_Headlines_Dataset.json"

# A small stopword list so "common words" shows content words, not "the/a/of".
STOPWORDS = set("""a an the of to in on at for and or but is are was were be been
being this that these those it its as with by from he she they them his her their
we you your i my me us our do does did has have had not no s""".split())


def load():
    records = [json.loads(line) for line in open(DATA)]
    df = pd.DataFrame(records)
    df["label"] = df["is_sarcastic"].map({1: "sarcastic", 0: "genuine"})
    return df


def basic_stats(df):
    print("=" * 60)
    print(f"Total headlines: {len(df)}")
    print(df["label"].value_counts())

    # --- length in words and characters ---
    df["n_words"] = df["headline"].str.split().str.len()
    df["n_chars"] = df["headline"].str.len()

    print("\nMean length by class:")
    print(df.groupby("label")[["n_words", "n_chars"]].mean().round(2))

    # --- punctuation: Onion headlines are famously "straight" AP-style;
    #     HuffPost uses more question marks, quotes, numbers, etc. ---
    df["has_question"] = df["headline"].str.contains(r"\?")
    df["has_number"]   = df["headline"].str.contains(r"\d")
    df["has_quote"]    = df["headline"].str.contains(r"['\"]")
    print("\nShare of headlines containing... (by class)")
    print(df.groupby("label")[["has_question", "has_number", "has_quote"]]
            .mean().round(3))
    return df


def top_words(df):
    def word_counts(subset):
        c = Counter()
        for h in subset["headline"]:
            for w in re.findall(r"[a-z']+", h.lower()):
                if w not in STOPWORDS and len(w) > 2:
                    c[w] += 1
        return c

    sarc = word_counts(df[df.is_sarcastic == 1])
    real = word_counts(df[df.is_sarcastic == 0])
    print("\nTop 15 content words -- SARCASTIC (Onion):")
    print([w for w, _ in sarc.most_common(15)])
    print("\nTop 15 content words -- GENUINE (HuffPost):")
    print([w for w, _ in real.most_common(15)])
    return sarc, real


# ----------------------------------------------------------------------
# Chart 1: length distribution by class
# ----------------------------------------------------------------------
def chart_length(df):
    plt.figure(figsize=(9, 5))
    bins = np.arange(0, 30, 1)
    for label, color in [("genuine", "#3498db"), ("sarcastic", "#e74c3c")]:
        plt.hist(df[df.label == label]["n_words"], bins=bins, alpha=0.6,
                 label=label, color=color, density=True)
    plt.title("Headline length (words) by class")
    plt.xlabel("words per headline")
    plt.ylabel("proportion of headlines")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart1_length.png", dpi=120)
    plt.close()
    print("Saved chart1_length.png")


# ----------------------------------------------------------------------
# Chart 2: punctuation / number usage by class
# ----------------------------------------------------------------------
def chart_punct(df):
    feats = ["has_question", "has_number", "has_quote"]
    labels = ["genuine", "sarcastic"]
    means = df.groupby("label")[feats].mean()
    x = np.arange(len(feats))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, means.loc["genuine"], w, label="genuine", color="#3498db")
    plt.bar(x + w/2, means.loc["sarcastic"], w, label="sarcastic", color="#e74c3c")
    plt.xticks(x, ["contains '?'", "contains a number", "contains a quote"])
    plt.ylabel("share of headlines")
    plt.title("Punctuation & number usage: genuine vs sarcastic")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart2_punctuation.png", dpi=120)
    plt.close()
    print("Saved chart2_punctuation.png")


# ----------------------------------------------------------------------
# Chart 3: words most distinctive of each class
#   ratio = P(word | sarcastic) / P(word | genuine)
# ----------------------------------------------------------------------
def chart_distinctive(sarc, real):
    total_s, total_r = sum(sarc.values()), sum(real.values())
    common = {w for w in sarc if sarc[w] >= 40} | {w for w in real if real[w] >= 40}
    rows = []
    for w in common:
        ps = (sarc[w] + 1) / total_s
        pr = (real[w] + 1) / total_r
        rows.append((w, np.log2(ps / pr)))
    rows.sort(key=lambda r: r[1])
    most_genuine = rows[:12]      # strongly HuffPost
    most_sarcastic = rows[-12:]   # strongly Onion

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh([w for w, _ in most_sarcastic], [v for _, v in most_sarcastic],
                 color="#e74c3c")
    axes[0].set_title("Words most skewed toward SARCASTIC (Onion)")
    axes[0].set_xlabel("log2( P(word|sarc) / P(word|genuine) )")
    axes[1].barh([w for w, _ in most_genuine], [v for _, v in most_genuine],
                 color="#3498db")
    axes[1].set_title("Words most skewed toward GENUINE (HuffPost)")
    axes[1].set_xlabel("log2 odds ratio")
    plt.tight_layout()
    plt.savefig("chart3_distinctive_words.png", dpi=120)
    plt.close()
    print("Saved chart3_distinctive_words.png")
    print("\nMost 'Onion' words:", [w for w, _ in reversed(most_sarcastic)])
    print("Most 'HuffPost' words:", [w for w, _ in most_genuine])


if __name__ == "__main__":
    df = load()
    df = basic_stats(df)
    sarc, real = top_words(df)
    chart_length(df)
    chart_punct(df)
    chart_distinctive(sarc, real)
    print("\nDone.")
