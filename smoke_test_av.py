"""Compatibility launcher; implementation lives in scripts/evaluation/smoke_test_av.py."""
from pathlib import Path
import os
import runpy

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_path(str(ROOT / "scripts/evaluation/smoke_test_av.py"), run_name="__main__")
