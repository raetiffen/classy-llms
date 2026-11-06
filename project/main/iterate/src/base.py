"""
    Basic approach to iteration
    First, classifies with the initial policy. Then, gives a LLM the policy and list of false positives/negatives, asking it to update the policy. Runs the new policy, and keeps it if the F1 score improves. Runs for a user-set number of iterations, or until in-sample F1 == 1.

    INPUT:
        - policy_init.txt, initial policy text
        - labels_train.csv, columns: "id", "content", "gold"
    OUTPUT:
        - policy_final.txt, final policy text
        - stats.csv, stats from iterations
"""
import sys
import ollama
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from tasks import classify, evaluate

ITERATION_COUNT = 4

def upd_policy(policy, merged):
    prompt = "Your job is to refine a LLM classifier policy. Below is the current policy, list of false positives, and list of false negatives. Return solely an updated version of the policy to try and catch false negatives and not mark the false positives. Keep in mind overfitting concerns and concision (STRICT policy limit to 500 tokens).\n"
    prompt += "CURRENT POLICY TEXT:\n" + policy + "\n\n"
    prompt += "FALSE POSITIVES:\n-----\n"
    for content in merged.loc[(merged["label"] == 1) & (merged["gold"] == 0), "content"]:
        prompt += f"{content}\n-----\n"
    prompt += "\nFALSE NEGATIVES:\n-----\n"
    for content in merged.loc[(merged["label"] == 0) & (merged["gold"] == 1), "content"]:
        prompt += f"{content}\n-----\n"
    prompt += "REFINED POLICY:"
    response = ollama.chat(
        model="qwen3:14b",
        messages=[{"role": "user", "content": prompt}],
        think=True,
        options={"num_ctx": 32768, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
        keep_alive=0,
    )
    return response.message.content.strip()

# Read inputs
labels_train = pd.read_csv(Path(__file__).parents[1] / "input" / "labels_train.csv")
policy = (Path(__file__).parents[1] / "input" / "policy_init.txt").read_text(encoding="utf-8")

stats = pd.DataFrame(columns=["iteration", "precision", "recall", "f1", "length", "accepted"])
best_policy, best_f1, best_merged = None, -1, None

for i in range(ITERATION_COUNT):
    if best_f1 != 1:
        print(f"Iteration {i}")

        # Classify and evaluate policy
        labeled = classify(labels_train[["id", "content"]], policy)["labeled.csv"]
        merged = labels_train.merge(labeled, on="id")
        p, r, f = map(float, evaluate(merged[["gold", "label"]])["stats.txt"].strip("[]\n").split(","))

        # Keep policy only if F1 improves or remains stable
        accepted = f >= best_f1
        if accepted:
            best_policy, best_f1, best_merged = policy, f, merged
        stats.loc[len(stats)] = [i, p, r, f, len(policy.split()), accepted]
        print(f"precision: {p:.3f}  recall: {r:.3f}  f1: {f:.3f}  {'accepted' if accepted else 'rejected'}")

        # Refine from best policy (skip on last iteration)
        if i < ITERATION_COUNT - 1 and best_f1 != 1:
            policy = upd_policy(best_policy, best_merged)

# Output
(Path(__file__).parents[1] / "output" / "policy_final.txt").write_text(best_policy, encoding="utf-8")
stats.to_csv(Path(__file__).parents[1] / "output" / "stats.csv", index=False)
