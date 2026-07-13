from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MiningUIResult:
  mining_active: bool
  progress_pct: float | None
  interaction_hint: bool
  has_label: bool = False


class MiningScreenUI:
  """Detecta barra 'Mining ore' e area de interacao na parte inferior da tela."""

  def __init__(
    self,
    *,
    progress_roi: dict[str, int],
    white_threshold: int = 175,
    min_bar_pixels: int = 120,
    min_progress_width_px: int = 40,
    requires_label: bool = True,
    max_bar_height_px: int = 14,
    max_bar_width_ratio: float = 0.88,
  ) -> None:
    self.progress_roi = dict(progress_roi)
    self.white_threshold = white_threshold
    self.min_bar_pixels = min_bar_pixels
    self.min_progress_width_px = min_progress_width_px
    self.requires_label = requires_label
    self.max_bar_height_px = max_bar_height_px
    self.max_bar_width_ratio = max_bar_width_ratio

  def _crop(self, frame_bgr: np.ndarray) -> np.ndarray:
    roi = self.progress_roi
    x1 = int(roi["left"])
    y1 = int(roi["top"])
    x2 = x1 + int(roi["width"])
    y2 = y1 + int(roi["height"])
    return frame_bgr[y1:y2, x1:x2]

  def detect_crop(self, crop_bgr: np.ndarray) -> MiningUIResult:
    """Detecta na imagem ja recortada (ROI da barra)."""
    if crop_bgr.size == 0:
      return MiningUIResult(False, None, False, False)
    return self._detect_on_crop(crop_bgr)

  def detect(self, frame_bgr: np.ndarray) -> MiningUIResult:
    crop = self._crop(frame_bgr)
    if crop.size == 0:
      return MiningUIResult(False, None, False, False)
    return self._detect_on_crop(crop)

  def _detect_on_crop(self, crop: np.ndarray) -> MiningUIResult:
    h, w = crop.shape[:2]
    if h < 12 or w < 40:
      return MiningUIResult(False, None, False, False)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white = gray >= self.white_threshold
    white_pixels = int(np.count_nonzero(white))
    if white_pixels < self.min_bar_pixels:
      return MiningUIResult(False, None, white_pixels > 30, False)

    text_h = max(1, h // 3)
    text_band = white[:text_h, :]
    text_px = int(np.count_nonzero(text_band))
    text_density = text_px / max(text_band.size, 1)
    has_label = 0.015 <= text_density <= 0.35 and text_px >= 25

    bar_area = white[text_h:, :]
    if bar_area.size == 0:
      return MiningUIResult(False, None, has_label, has_label)

    row_scores = np.sum(bar_area, axis=1)
    if int(row_scores.max()) < self.min_progress_width_px:
      return MiningUIResult(False, None, has_label, has_label)

    best_row = int(np.argmax(row_scores))
    row = bar_area[best_row]
    cols = np.where(row)[0]
    if cols.size < self.min_progress_width_px:
      return MiningUIResult(False, None, has_label, has_label)

    left = int(cols[0])
    right = int(cols[-1])
    bar_width = max(right - left + 1, 1)
    if bar_width > w * self.max_bar_width_ratio:
      return MiningUIResult(False, None, has_label, has_label)

    threshold = self.min_progress_width_px * 0.45
    band_start = best_row
    band_end = best_row
    while band_start > 0 and row_scores[band_start - 1] >= threshold:
      band_start -= 1
    while band_end < len(row_scores) - 1 and row_scores[band_end + 1] >= threshold:
      band_end += 1
    if band_end - band_start + 1 > self.max_bar_height_px:
      return MiningUIResult(False, None, has_label, has_label)

    filled = 0
    for col in range(left, right + 1):
      if row[col]:
        filled += 1
    progress = float(np.clip(100.0 * filled / bar_width, 0.0, 100.0))
    label_ok = has_label if self.requires_label else True
    mining_active = label_ok and 5.0 < progress < 98.0
    return MiningUIResult(mining_active, progress, has_label, has_label)

  def debug_crop(
    self,
    crop_bgr: np.ndarray,
    result: MiningUIResult,
    *,
    status_text: str | None = None,
  ) -> np.ndarray:
    """Overlay na imagem ja recortada."""
    crop = crop_bgr.copy()
    if crop.size == 0:
      return np.zeros((36, 200, 3), dtype=np.uint8)
    if status_text is not None:
      txt = status_text
      low = status_text.lower()
      if " ok" in low or low.startswith("mining ore"):
        color = (0, 255, 0)
      elif "~" in status_text:
        color = (0, 220, 255)
      else:
        color = (0, 180, 255)
    else:
      color = (0, 255, 0) if result.mining_active else (0, 180, 255)
      txt = "mining" if result.mining_active else "idle"
      if result.has_label:
        txt += " label"
      if result.progress_pct is not None:
        txt += f" {result.progress_pct:.0f}%"
    cv2.putText(
      crop,
      txt,
      (6, crop.shape[0] - 8),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.45,
      color,
      1,
      cv2.LINE_AA,
    )
    return crop
