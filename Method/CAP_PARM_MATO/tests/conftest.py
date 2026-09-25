from __future__ import annotations

import sys
from pathlib import Path


CAP_DIR = Path(__file__).resolve().parents[1]
if str(CAP_DIR) not in sys.path:
    sys.path.insert(0, str(CAP_DIR))
