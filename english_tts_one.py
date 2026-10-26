"""Compatibility launcher; implementation lives in scripts/tts/english_tts_one.py."""
from pathlib import Path
import os
import runpy

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
runpy.run_path(str(ROOT / "scripts/tts/english_tts_one.py"), run_name="__main__")
