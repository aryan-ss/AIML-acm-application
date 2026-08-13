"""
Task 1: Spotify Track Popularity Analysis
=========================================
Goal: load and clean the Spotify Tracks dataset, then explore which audio
features relate to a track's popularity through a few visualizations.

Dataset: Spotify Tracks Dataset (maharshipandya) - 114,000 tracks, one row each,
with audio features (danceability, energy, tempo, ...) and a popularity score 0-100.

Run:  python3 spotify_analysis.py
Outputs: prints a cleaning report + findings, and saves charts as PNGs.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to file without needing a display
import matplotlib.pyplot as plt

DATA = "spotify_tracks.csv"


# ----------------------------------------------------------------------
# 1. Load + basic cleaning
# ----------------------------------------------------------------------
def load_and_clean():
    df = pd.read_csv(DATA)
    print("=" * 60)
    print("RAW SHAPE:", df.shape)
    print("\nColumns & dtypes:")
    print(df.dtypes)

    # The first column is an unnamed row index left over from the export.
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    # --- missing values ---
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  (none)")

    # A handful of rows have a null artist/album/track name. They are useless
    # for analysis (no track identity), and there are very few, so we drop them.
    before = len(df)
    df = df.dropna(subset=["track_name", "artists", "album_name"])
    print(f"\nDropped {before - len(df)} rows with null text fields.")

    # --- duplicates ---
    # The same song can appear under multiple genres. For a per-track view of
    # popularity we de-duplicate on track_id, keeping the first genre listing.
    before = len(df)
    df_unique = df.drop_duplicates(subset="track_id")
    print(f"Duplicate track_ids collapsed: {before - len(df_unique)} "
          f"rows removed ({len(df_unique)} unique tracks).")

    return df, df_unique  # df keeps genre rows; df_unique is per-track


# ----------------------------------------------------------------------
# 2. Which numeric features correlate with popularity?
# ----------------------------------------------------------------------
def correlation_chart(df_unique):
    numeric = ["danceability", "energy", "loudness", "speechiness",
               "acousticness", "instrumentalness", "liveness", "valence",
               "tempo", "duration_ms", "explicit"]
    # explicit is bool -> make it numeric so it joins the correlation
    d = df_unique.copy()
    d["explicit"] = d["explicit"].astype(int)

    corr = d[numeric + ["popularity"]].corr()["popularity"].drop("popularity")
    corr = corr.sort_values()

    print("\n" + "=" * 60)
    print("Correlation of each feature with popularity:")
    print(corr.round(3))

    plt.figure(figsize=(8, 5))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr.values]
    plt.barh(corr.index, corr.values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Correlation of audio features with track popularity")
    plt.xlabel("Pearson correlation with popularity")
    plt.tight_layout()
    plt.savefig("chart1_correlations.png", dpi=120)
    plt.close()
    print("Saved chart1_correlations.png")
    return corr


# ----------------------------------------------------------------------
# 3. Feature vs popularity - binned means (clearer than a raw scatter
#    on 100k points, which is just a cloud)
# ----------------------------------------------------------------------
def binned_feature_plot(df_unique):
    features = ["danceability", "energy", "loudness", "acousticness"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), features):
        # cut the feature into 20 bins, average popularity within each bin
        bins = pd.cut(df_unique[feat], 20)
        grouped = df_unique.groupby(bins, observed=True)["popularity"].mean()
        centers = [interval.mid for interval in grouped.index]
        ax.plot(centers, grouped.values, marker="o", color="#3498db")
        ax.set_title(f"Mean popularity vs {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("mean popularity")
        ax.grid(alpha=0.3)
    plt.suptitle("How average popularity changes across each feature", y=1.02)
    plt.tight_layout()
    plt.savefig("chart2_binned_features.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved chart2_binned_features.png")


# ----------------------------------------------------------------------
# 4. Popularity by genre (uses the genre-level rows, not de-duplicated)
# ----------------------------------------------------------------------
def genre_plot(df):
    g = df.groupby("track_genre")["popularity"].mean().sort_values()
    top = g.tail(15)
    bottom = g.head(15)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(top.index, top.values, color="#2ecc71")
    axes[0].set_title("15 most popular genres (mean popularity)")
    axes[0].set_xlabel("mean popularity")
    axes[1].barh(bottom.index, bottom.values, color="#e74c3c")
    axes[1].set_title("15 least popular genres (mean popularity)")
    axes[1].set_xlabel("mean popularity")
    plt.tight_layout()
    plt.savefig("chart3_genres.png", dpi=120)
    plt.close()
    print("Saved chart3_genres.png")
    print("\nTop 5 genres by mean popularity:")
    print(g.tail(5).round(1))
    print("\nBottom 5 genres by mean popularity:")
    print(g.head(5).round(1))


# ----------------------------------------------------------------------
# 5. Popularity distribution (context: how is the target spread out?)
# ----------------------------------------------------------------------
def popularity_hist(df_unique):
    plt.figure(figsize=(8, 5))
    plt.hist(df_unique["popularity"], bins=50, color="#9b59b6",
             edgecolor="white")
    plt.title("Distribution of track popularity")
    plt.xlabel("popularity (0-100)")
    plt.ylabel("number of tracks")
    zero = (df_unique["popularity"] == 0).mean() * 100
    plt.axvline(df_unique["popularity"].mean(), color="black",
                linestyle="--", label=f"mean = {df_unique['popularity'].mean():.1f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart4_popularity_dist.png", dpi=120)
    plt.close()
    print("Saved chart4_popularity_dist.png")
    print(f"\nShare of tracks with popularity == 0: {zero:.1f}%")


if __name__ == "__main__":
    df, df_unique = load_and_clean()
    corr = correlation_chart(df_unique)
    binned_feature_plot(df_unique)
    genre_plot(df)
    popularity_hist(df_unique)

    print("\n" + "=" * 60)
    print("DONE. Strongest positive / negative correlates with popularity:")
    print(f"  most positive: {corr.idxmax()} ({corr.max():+.3f})")
    print(f"  most negative: {corr.idxmin()} ({corr.min():+.3f})")
