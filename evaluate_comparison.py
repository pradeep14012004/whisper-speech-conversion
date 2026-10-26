"""Compatibility launcher for scripts/evaluation/evaluate_comparison.py."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

runpy.run_path(
    str(ROOT / "scripts" / "evaluation" / "evaluate_comparison.py"),
    run_name="__main__",
)
