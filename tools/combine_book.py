"""Uninstalled entry: python tools/combine_book.py <topic>."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_tools = str(_TOOLS)
if _tools not in sys.path:
    sys.path.insert(0, _tools)

from book_combiner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
