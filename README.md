# AI/ML Club Recruitment Tasks — Aadya Jain

Solutions to the three recruitment tasks. Each folder is self-contained with its code, charts, and a `README.md` holding the written answer.

| Task | Folder | What it does |
|---|---|---|
| 1. Spotify Track Popularity | [`task1-spotify/`](task1-spotify) | Clean 114k tracks; explore which audio features relate to popularity (4 charts + written answer). |
| 2. Sarcasm Headline Analysis | [`task2-sarcasm/`](task2-sarcasm) | Compare Onion (sarcastic) vs HuffPost (genuine) headlines — length, punctuation, distinctive words (3 charts + written answer). |
| 3. Cat vs. Dog Classifier | [`task3-catdog/`](task3-catdog) | MobileNetV2 transfer-learning notebook; accuracy + 5 example predictions. |

## Running
Tasks 1 & 2 are plain scripts (pandas + matplotlib):
```bash
cd task1-spotify && python3 spotify_analysis.py
cd task2-sarcasm && python3 sarcasm_analysis.py
```
Task 3 is a Jupyter notebook using TensorFlow — see [`task3-catdog/README.md`](task3-catdog/README.md).

## Datasets
Not committed (see `.gitignore`) to keep the repo small. Download links are in each task's README.
