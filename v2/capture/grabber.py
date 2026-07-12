"""
Captura MSS — minimap + HUD (barra Mining ore via get_hud_roi).
Não interpreta pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mss
import numpy as np

from v2.core.config import get_hud_roi, get_minimap_roi
from v2.ore_roi_calibrate import screen_size


@dataclass
class Grabber:
  cfg: dict[str, Any]
  _sct: mss.mss | None = None

  def __enter__(self) -> Grabber:
    self._sct = mss.mss()
    return self

  def __exit__(self, *_) -> None:
    if self._sct is not None:
      self._sct.close()
      self._sct = None

  def grab(self) -> tuple[np.ndarray, np.ndarray]:
    if self._sct is None:
      raise RuntimeError("Grabber fora de context manager.")
    mini = np.array(self._sct.grab(get_minimap_roi(self.cfg)))[:, :, :3]
    hud = np.array(self._sct.grab(get_hud_roi(self.cfg)))[:, :, :3]
    return mini, hud

  def grab_screen(self) -> np.ndarray:
    """Frame full-resolution (para calibrar ROI Mining ore)."""
    if self._sct is None:
      raise RuntimeError("Grabber fora de context manager.")
    w, h = screen_size(self.cfg)
    return np.array(self._sct.grab({"left": 0, "top": 0, "width": w, "height": h}))[:, :, :3]
