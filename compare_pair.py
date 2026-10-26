"""Compatibility launcher; implementation lives in scripts/evaluation/compare_pair.py."""
from pathlib import Path
import os
import runpy

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_path(str(ROOT / "scripts/evaluation/compare_pair.py"), run_name="__main__")
