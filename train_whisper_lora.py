"""Compatibility launcher for scripts/training/train_whisper_lora.py."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

runpy.run_path(
    str(ROOT / "scripts" / "training" / "train_whisper_lora.py"),
    run_name="__main__",
)
