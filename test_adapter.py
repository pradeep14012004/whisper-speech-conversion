"""Compatibility launcher; implementation lives in scripts/evaluation/test_adapter.py."""
from pathlib import Path
import os
import runpy

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_path(str(ROOT / "scripts/evaluation/test_adapter.py"), run_name="__main__")
