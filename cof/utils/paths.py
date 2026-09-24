"""Define the canonical project paths used by command-line workflows.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yml"
USER_CONFIG_PATH = PROJECT_ROOT / ".config" / "cof.yml"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
