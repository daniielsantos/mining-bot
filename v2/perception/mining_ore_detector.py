"""
Detecta a barra \"Mining ore\" (label branco em fundo escuro) via template match.

Usado no FINAL_APPROACH apos tap E: se a UI aparecer, estamos no range de minerar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_V2_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_TEMPLATE = _V2_DIR / "assets" / "mining_ore_label.png"


@dataclass(frozen=True)
class MiningOreHit:
  found: bool
  score: float
  scale: float | None = None
  loc: tuple[int, int] | None = None
  raw_hit: bool = False
  near_miss: bool = False


class MiningOreDetector:
  def __init__(self, cfg: dict[str, Any]) -> None:
    nav = cfg.get("navigation", {})
    ore = nav.get("mining_ore", cfg.get("mining_ore", {}))
    self.threshold = float(ore.get("match_threshold", 0.85))
    self.confirm_frames = max(1, int(ore.get("confirm_frames", 2)))
    # Quao perto do limiar para log/overlay \"quase\".
    self.near_gap = float(ore.get("near_miss_gap", 0.08))
    scales = ore.get(
      "scales",
      [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0],
    )
    self.scales = tuple(float(s) for s in scales)
    # Valida patch: texto claro em fundo escuro (rejeita falso positivo em terreno).
    self.require_ui_contrast = bool(ore.get("require_ui_contrast", True))
    self.contrast_dark_max = float(ore.get("contrast_dark_max", 78))
    self.contrast_bright_min = float(ore.get("contrast_bright_min", 100))
    self.contrast_gap_min = float(ore.get("contrast_gap_min", 80))
    # Contraste abaixo deste score bruto ainda é checado (ambient ~0.55–0.65).
    self.contrast_check_floor = float(ore.get("contrast_check_floor", 0.30))
    # HUD vazio / terreno: TM_CCOEFF ~0.60 sem texto — cap reportado <0.40.
    self.contrast_fail_scale = float(ore.get("contrast_fail_scale", 0.50))
    self.contrast_fail_cap = float(ore.get("contrast_fail_cap", 0.35))
    # Present floor (overlay / READY) — alias hold_min.
    if "hold_min" in ore:
      self.hold_min = float(ore["hold_min"])
    elif "present_threshold" in ore:
      self.hold_min = float(ore["present_threshold"])
    else:
      self.hold_min = float(self.threshold)

    tmpl = Path(str(ore.get("template", _DEFAULT_TEMPLATE)))
    if not tmpl.is_file():
      alt = _V2_DIR / tmpl
      tmpl = alt if alt.is_file() else _DEFAULT_TEMPLATE
    self.template_path = tmpl

    self._tmpl_gray: np.ndarray | None = None
    self._ok_streak = 0
    self._last_score = 0.0
    self._last_hit = MiningOreHit(False, 0.0)

  def reset(self) -> None:
    self._ok_streak = 0
    self._last_score = 0.0
    self._last_hit = MiningOreHit(False, 0.0)

  @property
  def last_score(self) -> float:
    return float(self._last_score)

  @property
  def last_hit(self) -> MiningOreHit:
    return self._last_hit

  def _ensure_template(self) -> bool:
    if self._tmpl_gray is not None:
      return True
    if not self.template_path.is_file():
      return False
    img = cv2.imread(str(self.template_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
      return False
    self._tmpl_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return True

  def _ui_contrast_ok(self, hay: np.ndarray, loc: tuple[int, int], tw: int, th: int) -> bool:
    x, y = loc
    if y < 0 or x < 0 or y + th > hay.shape[0] or x + tw > hay.shape[1]:
      return False
    patch = hay[y : y + th, x : x + tw]
    if patch.size < 40:
      return False
    flat = np.sort(patch.reshape(-1).astype(np.float32))
    n = len(flat)
    dark = float(np.mean(flat[: max(1, n // 2)]))
    bright = float(np.mean(flat[int(n * 0.75) :]))
    return (
      bright >= self.contrast_bright_min
      and dark <= self.contrast_dark_max
      and (bright - dark) >= self.contrast_gap_min
    )

  def detect(self, crop_bgr: np.ndarray | None) -> MiningOreHit:
    """Match no crop ja recortado (HUD / mining_ore ROI)."""
    if not self._ensure_template() or crop_bgr is None or crop_bgr.size == 0:
      self._last_score = 0.0
      self._last_hit = MiningOreHit(False, 0.0)
      return self._last_hit

    if crop_bgr.ndim == 2:
      hay = crop_bgr
    else:
      hay = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    tmpl = self._tmpl_gray
    assert tmpl is not None
    best = -1.0
    best_scale: float | None = None
    best_loc: tuple[int, int] | None = None
    best_wh: tuple[int, int] | None = None
    th0, tw0 = tmpl.shape[:2]

    for scale in self.scales:
      tw = max(16, int(round(tw0 * scale)))
      th = max(8, int(round(th0 * scale)))
      if th >= hay.shape[0] or tw >= hay.shape[1]:
        continue
      resized = cv2.resize(
        tmpl,
        (tw, th),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
      )
      res = cv2.matchTemplate(hay, resized, cv2.TM_CCOEFF_NORMED)
      _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
      score = float(max_v)
      if score > best:
        best = score
        best_scale = float(scale)
        best_loc = (int(max_l[0]), int(max_l[1]))
        best_wh = (tw, th)

    contrast_ok = True
    if (
      self.require_ui_contrast
      and best_loc is not None
      and best_wh is not None
      and best >= self.contrast_check_floor
    ):
      contrast_ok = self._ui_contrast_ok(hay, best_loc, best_wh[0], best_wh[1])

    # Sem contraste UI: não reportar score ambient (~0.60) como quase-match.
    if self.require_ui_contrast and not contrast_ok and best > 0.0:
      reported = min(best * self.contrast_fail_scale, self.contrast_fail_cap)
    else:
      reported = best
    self._last_score = max(0.0, float(reported))

    raw_hit = best >= self.threshold and contrast_ok
    if raw_hit:
      self._ok_streak += 1
    else:
      self._ok_streak = 0

    found = self._ok_streak >= self.confirm_frames
    near_miss = (
      (not found)
      and (not raw_hit)
      and self._last_score >= max(0.0, self.threshold - self.near_gap)
    )
    self._last_hit = MiningOreHit(
      found=found,
      score=self._last_score,
      scale=best_scale,
      loc=best_loc,
      raw_hit=raw_hit,
      near_miss=near_miss,
    )
    return self._last_hit
