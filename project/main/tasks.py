"""
    Shared module for running tasks
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd

def run_task(task, inputs, script=None):
    # Clean input/output folder
    for i in ("input", "output"):
        for f in (Path(__file__).parent / task / i).iterdir():
            f.unlink()
    
    # Plop inputs into input folder
    for name, obj in inputs.items():
        path = Path(__file__).parent / task / "input" / name
        # If df, make csv
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False)
        # Else, it's text
        else:
            path.write_text(obj, encoding="utf-8")

    # Run the .py in src
    subprocess.run([sys.executable, Path(__file__).parent / task / "src" / f"{script or task}.py"], check=True)

    # Return output
    return {f.name: pd.read_csv(f) if f.suffix == ".csv" else f.read_text(encoding="utf-8") for f in (Path(__file__).parent / task / "output").iterdir()}

def classify(content, policy):
    return run_task("classify", {"content.csv": content, "policy.txt": policy})

def iterate(labels_train, policy, method):
    return run_task("iterate", {"labels_train.csv": labels_train, "policy_init.txt": policy}, method)

def evaluate(labels):
    return run_task("evaluate", {"labels.csv": labels})