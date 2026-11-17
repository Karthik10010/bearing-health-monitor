"""Rebuild everything from scratch: data -> features -> models -> edge -> RL -> figures -> tests.

    .venv\\Scripts\\python.exe scripts\\run_all.py            (about 20-30 minutes on a laptop CPU)
    .venv\\Scripts\\python.exe scripts\\run_all.py --skip-rl  (keep the existing PPO agent)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_download_data.py", "02_build_features.py", "03_train_classifier.py", "04_edge_benchmark.py",
         "05_train_rl.py", "06_evaluate_rl.py", "07_make_figures.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rl", action="store_true", help="skip PPO training (05)")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    for step in STEPS:
        if a.skip_rl and step == "05_train_rl.py":
            continue
        t = time.time()
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / step)], check=True)
        print(f"=== {step} done in {time.time() - t:.0f}s ===", flush=True)
    if not a.no_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, cwd=HERE.parent)
