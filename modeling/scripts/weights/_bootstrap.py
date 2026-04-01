from __future__ import annotations

import sys
from pathlib import Path


def ensure_modeling_root_on_path() -> Path:
    modeling_root = Path(__file__).resolve().parents[2]
    modeling_root_str = str(modeling_root)
    if modeling_root_str not in sys.path:
        sys.path.insert(0, modeling_root_str)
    return modeling_root
