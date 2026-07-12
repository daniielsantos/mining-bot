"""Walker v2 — Facing Pursuit (teclas seguradas + feedback de progresso)."""

from __future__ import annotations

from typing import Any

from v2.navigation.facing_walker import FacingWalkController

# Legado — PursuitWalker ainda disponível se precisar comparar.
AlignWalkController = FacingWalkController


def build_walker_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
  del cfg
  return {}
