"""Compatibility launcher for scripts/evaluation/eval_large_v3.py."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

runpy.run_path(
    str(ROOT / "scripts" / "evaluation" / "eval_large_v3.py"),
    run_name="__main__",
)
