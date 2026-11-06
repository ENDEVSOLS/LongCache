"""Backward-compatible delegating wrapper for scripts/verify_release_v1.py."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_release_v1 import main, verify_release, _source_hash

if __name__ == "__main__":
    raise SystemExit(main())
