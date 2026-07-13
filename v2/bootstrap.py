"""Garante imports do pacote v2 — roda standalone (só a pasta v2/) ou de mining_bot/."""

from __future__ import annotations

import sys
from pathlib import Path

_V2_DIR = Path(__file__).resolve().parent


def setup() -> Path:
  """Coloca o diretório pai do pacote `v2` no sys.path."""
  parent = str(_V2_DIR.parent)
  if parent not in sys.path:
    sys.path.insert(0, parent)
  return _V2_DIR
