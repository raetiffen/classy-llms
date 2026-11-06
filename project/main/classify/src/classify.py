"""
    Classification

    INPUT:
        - policy.txt, policy text
        - content.csv, content to label, columns: "id", "content"
    OUTPUT:
        - labeled.csv, labeled content, columns: "id", "label"
"""
import ollama
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Read inputs
content = pd.read_csv(Path(__file__).parents[1] / "input" / "content.csv")
policy_text = (Path(__file__).parents[1] / "input" / "policy.txt").read_text(encoding="utf-8") + "\n## OUTPUT\n\nReturn exactly one character: 0 or 1.\nDo not include any explanation or punctuation.\n0 = Content does NOT violate this policy.\n1 = Content violates this policy."

labels = content[["id"]].assign(label="")

# Run prompts
for i in tqdm(labels["id"], desc="Classifying", leave=False):
    okay = False
    tries = 0
    while okay == False:
        label = ollama.chat(
            model="gpt-oss-safeguard:20b",
            messages=[
                {"role": "system", "content": policy_text },
                {"role": "user", "content": content.loc[content["id"] == i, "content"].item()},
            ],
            think="low" if tries == 0 else "medium",
            options={"num_ctx": 4096, "temperature": 0 if tries == 0 else 0.5}
        )
        if label.message.content.strip() == "0" or label.message.content.strip() == "1":
            okay = True
        tries += 1
        if tries == 3 and okay == False:
            raise RuntimeError(f"Failed to classify {i}: {label.message.content}")
    labels.loc[labels["id"] == i, "label"] = label.message.content.strip()

# Add labels to output folder
labels.to_csv(Path(__file__).parents[1] / "output" / "labeled.csv", index=False)

# Unload model
ollama.generate(model="gpt-oss-safeguard:20b", keep_alive=0)



"""
holding area

response = ollama.chat(
    model="qwen3:14b",
    messages=[{"role":user,content:"prompt"}],
    think=False,
    keep_alive = 0
)

response = ollama.chat(
    model="gpt-oss-safeguard:20b",
    messages=[
        {"role": "system", "content": policy_text},
        {"role": "user", "content": content_to_classify},
    ],
    think="low",
    keep_alive=0,
    options={"num_ctx": 8192, "temperature": 0},
)

label = response.message.content
reasoning = response.message.thinking
"""

