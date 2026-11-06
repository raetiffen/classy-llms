"""
    Evaluation

    INPUT:
        - labels.csv, two columns: "gold", "label"
    OUTPUT:
        - stats.txt: [precision, recall, f1]
"""
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# Read in labels
labels = pd.read_csv(Path(__file__).parents[1] / "input" / "labels.csv")
bad = labels[~labels["label"].isin([0, 1]) | ~labels["gold"].isin([0, 1])]
if len(bad) > 0:
    raise ValueError(f"Invalid labels:\n{bad}")

# Calculate scores
precision = precision_score(labels["gold"], labels["label"], pos_label=1, zero_division=0)
recall = recall_score(labels["gold"], labels["label"], pos_label=1, zero_division=0)
f1 = f1_score(labels["gold"], labels["label"], pos_label=1, zero_division=0)

# Output
(Path(__file__).parents[1] / "output" / "stats.txt").write_text(str([precision, recall, f1]))