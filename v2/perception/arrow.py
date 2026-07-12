"""Seta do jogador — wrapper minimap_tracker."""

from __future__ import annotations

from typing import Any

import numpy as np

from v2.core.types import ArrowState
from v2.navigation.bearing import FIRST_PERSON_FORWARD_DEG, first_person_tip


def detect_arrow(
  minimap_bgr: np.ndarray,
  tracker: Any,
  *,
  first_person: bool = False,
) -> tuple[ArrowState, Any]:
  """Retorna ArrowState + objeto legado (para walk_heading_from_arrow)."""
  result = tracker.detect(minimap_bgr)
  px, py = result.pivot()
  tip_x = result.arrow_tip_x
  tip_y = result.arrow_tip_y
  facing = result.arrow_angle_deg or tracker.last_facing_deg
  if first_person:
    # Seta fixa: so precisamos do pivô; frente = cima na tela.
    tip_x, tip_y = first_person_tip(float(px), float(py))
    facing = FIRST_PERSON_FORWARD_DEG
    try:
      result.arrow_tip_x = tip_x
      result.arrow_tip_y = tip_y
      result.arrow_angle_deg = facing
    except Exception:
      pass
  return (
    ArrowState(
      pivot_x=float(px),
      pivot_y=float(py),
      tip_x=tip_x,
      tip_y=tip_y,
      facing_deg=facing,
      detected=bool(result.arrow_detected) or first_person,
    ),
    result,
  )
