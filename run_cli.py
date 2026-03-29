from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEPS_DIR = PROJECT_ROOT / ".deps"
SRC_DIR = PROJECT_ROOT / "src"

if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from industrial_patch_inspector.cli import main


if __name__ == "__main__":
    main()
