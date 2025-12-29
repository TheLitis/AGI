"""
Pytest config.

Repo keeps modules as plain .py files in the repo root (env.py, trainer.py, ...).
Depending on how pytest is launched, the repo root may be missing from sys.path.
This file makes `pytest -q` work consistently.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

