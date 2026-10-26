"""Shared paths for scripts in this repository.

Keep this module at the repository root. Scripts can import ROOT and build
paths from it rather than depending on the shell's current working directory.

Example:
    from project_paths import ROOT, OUTPUTS
    manifest = OUTPUTS / "training_manifest.csv"

This helper does not automatically change existing scripts; migrate scripts
to these constants before moving them into subdirectories.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
ARCHIVE = ROOT / "archive"
CONVERSION_EXPERIMENTS = ARCHIVE / "conversion_experiments"


def project_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return ROOT.joinpath(*parts)


def output_path(*parts: str) -> Path:
    """Resolve a path under outputs/ and create its parent directory."""
    path = OUTPUTS.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
