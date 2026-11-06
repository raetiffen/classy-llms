"""
    Control script

    Available iteration methods:
        - "base": In each step, gives LLM a list of false positives/negatives and asks it to improve the policy
    
    INPUT:
        "policy_init.txt", initial policy
        "labels.csv", cols: "id", "content", "gold"
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from tasks import iterate, evaluate, classify


ITERATION_METHOD = "base"
TRAINTEST_SPLIT = 0.8 # proportion of dataset to include in training

def main():
    # Clear output folder for new run
    for f in (Path(__file__).parents[1] / "output").iterdir():
        f.unlink()

    # Prepare policy and ground truth data
    labels = pd.read_csv(Path(__file__).parents[1] / "input" / "labels.csv")
    policy_init = (Path(__file__).parents[1] / "input" / "policy_init.txt").read_text(encoding="utf-8")

    # Separate train/test
    labels = labels.sample(frac=1, random_state=1234)
    num_train = int(len(labels) * TRAINTEST_SPLIT)
    labels_train, labels_test = labels.iloc[:num_train], labels.iloc[num_train:]

    # Iterate policy on training data
    result = iterate(labels_train, policy_init, ITERATION_METHOD)
    policy_final, train_stats = result["policy_final.txt"], result["stats.csv"]
    (Path(__file__).parents[1] / "output" / "policy_final.txt").write_text(policy_final, encoding="utf-8")
    train_stats.to_csv(Path(__file__).parents[1] / "output" / "train_stats.csv", index=False)

    # Classify and evaluate original and final policies on test data
    initial_class = classify(labels_test[["id", "content"]], policy_init)["labeled.csv"]
    initial_stats = evaluate(labels_test[["id","gold"]].merge(initial_class, on="id"))["stats.txt"]
    final_class = classify(labels_test[["id", "content"]], policy_final)["labeled.csv"]
    final_stats = evaluate(labels_test[["id","gold"]].merge(final_class, on="id"))["stats.txt"]
    test_stats = pd.DataFrame(columns=["policy", "precision", "recall", "f1", "length"])
    for name, stats, policy in (("initial", initial_stats, policy_init), ("final", final_stats, policy_final)):
        p, r, f = map(float, stats.strip("[]\n").split(","))
        test_stats.loc[len(test_stats)] = [name, p, r, f, len(policy.split())]
        print(f"{name:<9} precision: {p:.3f}  recall: {r:.3f}  f1: {f:.3f}  length: {len(policy.split())}")
    test_stats.to_csv(Path(__file__).parents[1] / "output" / "test_stats.csv", index=False)


if __name__ == "__main__":
    main()