"""HUD de mineração."""

from __future__ import annotations

from typing import Any

import numpy as np
from v2.vendor.screen_ui import MiningUIResult

from v2.core.types import HudState


def detect_hud(hud_bgr: np.ndarray, ui: Any) -> tuple[HudState, MiningUIResult]:
  crop = ui.detect_crop(hud_bgr)
  return (
    HudState(
      mining_active=bool(crop.mining_active),
      progress_pct=crop.progress_pct,
      has_label=bool(crop.has_label),
    ),
    crop,
  )
