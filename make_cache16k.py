"""Compatibility launcher; implementation lives in scripts/data_preparation/make_cache16k.py."""
from pathlib import Path
import os
import runpy

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_path(str(ROOT / "scripts/data_preparation/make_cache16k.py"), run_name="__main__")
