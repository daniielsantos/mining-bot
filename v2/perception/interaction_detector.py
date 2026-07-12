"""
Detecta o prompt central via o glifo \"E\" no circulo (OCR-free).

Sinal primario: template match do E+circulo (asfalto + areia).
O texto INTERACTION e fraco/semi-transparente e so entra como bonus opcional.

Gates: limiar do E, match perto do centro do ROI, blob claro do E,
anel circular (Hough), e confirmacao multi-frame.
READY so com hit confirmado — sem timeout→READY / sem auto E.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_V2_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_E_ASPHALT = _V2_DIR / "assets" / "interaction_e_circle.png"
_DEFAULT_E_SAND = _V2_DIR / "assets" / "interaction_e_circle_sand.png"
_DEFAULT_TEXT = _V2_DIR / "assets" / "interaction_prompt.png"


@dataclass(frozen=True)
class InteractionHit:
  found: bool
  score: float
  scale: float | None = None
  loc: tuple[int, int] | None = None  # xy no crop (topo-esq do E)
  contrast_ok: bool = False  # blob claro do E
  center_ok: bool = False
  e_circle_ok: bool = False  # anel circular em torno do E
  text_score: float = 0.0  # bonus opcional (nao obrigatorio)


class InteractionDetector:
  def __init__(self, cfg: dict[str, Any]) -> None:
    nav = cfg.get("navigation", {})
    inter = nav.get("interaction", cfg.get("interaction", {}))
    self.threshold = float(
      inter.get(
        "match_threshold",
        nav.get("interaction_match_threshold", 0.72),
      )
    )
    self.confirm_frames = max(1, int(inter.get("confirm_frames", 5)))
    self.center_max_dx_frac = float(inter.get("center_max_dx_frac", 0.28))
    self.center_max_dy_frac = float(inter.get("center_max_dy_frac", 0.35))
    self.e_blob_min_abs = float(inter.get("e_blob_min_abs", 180.0))
    self.e_blob_min_local = float(inter.get("e_blob_min_local", 20.0))
    self.require_text = bool(inter.get("require_text", False))
    self.text_min_score = float(inter.get("text_min_score", 0.35))
    scales = inter.get(
      "scales",
      [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5],
    )
    self.scales = tuple(float(s) for s in scales)

    tmpl_paths = inter.get("e_templates")
    if isinstance(tmpl_paths, list) and tmpl_paths:
      paths = [Path(str(p)) for p in tmpl_paths]
    else:
      primary = Path(str(inter.get("template", _DEFAULT_E_ASPHALT)))
      secondary = Path(str(inter.get("template_sand", _DEFAULT_E_SAND)))
      paths = [primary, secondary]

    resolved: list[Path] = []
    for p in paths:
      if p.is_file():
        resolved.append(p)
        continue
      alt = _V2_DIR / p
      if alt.is_file():
        resolved.append(alt)
    if not resolved:
      for fallback in (_DEFAULT_E_ASPHALT, _DEFAULT_E_SAND):
        if fallback.is_file():
          resolved.append(fallback)
    self.template_paths = resolved

    text_path = Path(str(inter.get("text_template", _DEFAULT_TEXT)))
    if not text_path.is_file():
      alt = _V2_DIR / text_path
      text_path = alt if alt.is_file() else _DEFAULT_TEXT
    self.text_template_path = text_path

    self._e_tmpls_gray: list[np.ndarray] = []
    self._e_tmpls_pos: list[np.ndarray] = []
    self._text_gray: np.ndarray | None = None
    self._ok_streak = 0
    self._last_score = 0.0

  def reset(self) -> None:
    self._ok_streak = 0
    self._last_score = 0.0

  @staticmethod
  def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
      return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  @classmethod
  def _local_positive(cls, img: np.ndarray) -> np.ndarray:
    """Pixels localmente mais claros que o fundo (UI em areia clara)."""
    g = cls._to_gray(img).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 4.0)
    rel = np.clip(g - blur, 0.0, None)
    return cv2.normalize(rel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

  def _ensure_templates(self) -> bool:
    if self._e_tmpls_gray:
      return True
    for path in self.template_paths:
      img = cv2.imread(str(path), cv2.IMREAD_COLOR)
      if img is None or img.size == 0:
        continue
      g = self._to_gray(img)
      self._e_tmpls_gray.append(g)
      self._e_tmpls_pos.append(self._local_positive(img))
    if self.text_template_path.is_file():
      timg = cv2.imread(str(self.text_template_path), cv2.IMREAD_COLOR)
      if timg is not None and timg.size:
        # Metade direita = INTERACTION (sem o E).
        tw = timg.shape[1]
        self._text_gray = self._to_gray(timg[:, tw // 3 :])
    return bool(self._e_tmpls_gray)

  @property
  def last_score(self) -> float:
    return float(self._last_score)

  def _center_ok(
    self,
    gray_shape: tuple[int, int],
    loc: tuple[int, int],
    size: tuple[int, int],
  ) -> bool:
    h, w = gray_shape
    mx = loc[0] + size[0] * 0.5
    my = loc[1] + size[1] * 0.5
    cx = w * 0.5
    cy = h * 0.5
    # E fica a esquerda do bloco INTERACTION — permite deslocamento a esquerda.
    max_dx = w * self.center_max_dx_frac
    max_dy = h * self.center_max_dy_frac
    return (cx - max_dx) <= mx <= (cx + max_dx * 0.85) and abs(my - cy) <= max_dy

  def _bright_e_blob_ok(self, patch_bgr: np.ndarray) -> bool:
    """E solido branco (asfalto) ou realce local (areia)."""
    if patch_bgr.size == 0:
      return False
    g = self._to_gray(patch_bgr)
    h, w = g.shape
    cy, cx = h * 0.5, w * 0.5
    r = min(h, w) * 0.35
    yy, xx = np.ogrid[:h, :w]
    inner = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    if not np.any(inner):
      return False
    if float(g[inner].max()) >= self.e_blob_min_abs:
      return True
    pos = self._local_positive(patch_bgr)
    return float(pos[inner].mean()) >= self.e_blob_min_local

  def _ring_structure_ok(self, patch_bgr: np.ndarray) -> bool:
    """Exige anel circular em torno do E (rejeita letras brancas soltas)."""
    if patch_bgr.size == 0:
      return False
    g = self._to_gray(patch_bgr)
    h, w = g.shape
    if min(h, w) < 20:
      return False
    cy, cx = h * 0.5, w * 0.5
    r_max = min(h, w) * 0.5
    min_r = max(8, int(r_max * 0.52))
    max_r = max(min_r + 2, int(r_max * 0.98))
    center_tol = r_max * 0.28

    for src in (g, self._local_positive(patch_bgr)):
      blur = cv2.GaussianBlur(src, (3, 3), 0)
      circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(12, int(r_max)),
        param1=70,
        param2=16,
        minRadius=min_r,
        maxRadius=max_r,
      )
      if circles is None:
        continue
      for circ in circles[0]:
        dx = float(circ[0]) - cx
        dy = float(circ[1]) - cy
        if (dx * dx + dy * dy) ** 0.5 <= center_tol:
          return True
    return False

  def _match_e(
    self,
    crop_bgr: np.ndarray,
  ) -> tuple[float, float | None, tuple[int, int] | None, tuple[int, int] | None]:
    hay_gray = self._to_gray(crop_bgr)
    hay_pos = self._local_positive(crop_bgr)
    hay_maps = (hay_gray, hay_pos)

    best = -1.0
    best_scale: float | None = None
    best_loc: tuple[int, int] | None = None
    best_size: tuple[int, int] | None = None

    tmpl_pairs = list(zip(self._e_tmpls_gray, self._e_tmpls_pos))
    for tmpl_gray, tmpl_pos in tmpl_pairs:
      for tmpl in (tmpl_gray, tmpl_pos):
        th, tw = tmpl.shape[:2]
        for scale in self.scales:
          tw2 = max(14, int(round(tw * scale)))
          th2 = max(14, int(round(th * scale)))
          for hay in hay_maps:
            if th2 >= hay.shape[0] or tw2 >= hay.shape[1]:
              continue
            resized = cv2.resize(
              tmpl,
              (tw2, th2),
              interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            res = cv2.matchTemplate(hay, resized, cv2.TM_CCOEFF_NORMED)
            _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
            score = float(max_v)
            if score > best:
              best = score
              best_scale = float(scale)
              best_loc = (int(max_l[0]), int(max_l[1]))
              best_size = (tw2, th2)

    return best, best_scale, best_loc, best_size

  def _text_bonus(
    self,
    crop_bgr: np.ndarray,
    loc: tuple[int, int],
    size: tuple[int, int],
  ) -> float:
    """Match fraco de INTERACTION a direita do E (opcional)."""
    if self._text_gray is None:
      return 0.0
    x, y = loc
    ew, eh = size
    x1 = min(crop_bgr.shape[1], x + ew)
    x2 = min(crop_bgr.shape[1], x + ew + max(ew * 3, 80))
    y1 = max(0, y - 4)
    y2 = min(crop_bgr.shape[0], y + eh + 4)
    if x2 - x1 < 40 or y2 - y1 < 12:
      return 0.0
    region = self._to_gray(crop_bgr[y1:y2, x1:x2])
    tmpl = self._text_gray
    best = 0.0
    for scale in (0.85, 1.0, 1.15):
      tw = max(20, int(round(tmpl.shape[1] * scale)))
      th = max(10, int(round(tmpl.shape[0] * scale)))
      if th >= region.shape[0] or tw >= region.shape[1]:
        continue
      resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
      res = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
      best = max(best, float(cv2.minMaxLoc(res)[1]))
    return best

  def detect(self, crop_bgr: np.ndarray | None) -> InteractionHit:
    """Match no crop ja recortado (ROI de interacao)."""
    if not self._ensure_templates() or crop_bgr is None or crop_bgr.size == 0:
      self._last_score = 0.0
      return InteractionHit(False, 0.0)

    if crop_bgr.ndim == 2:
      crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)

    best, best_scale, best_loc, best_size = self._match_e(crop_bgr)
    self._last_score = max(0.0, best)

    contrast_ok = False
    center_ok = False
    e_circle_ok = False
    text_score = 0.0
    if best_loc is not None and best_size is not None and best >= 0:
      x, y = best_loc
      tw2, th2 = best_size
      patch = crop_bgr[y : y + th2, x : x + tw2]
      if patch.shape[0] == th2 and patch.shape[1] == tw2:
        contrast_ok = self._bright_e_blob_ok(patch)
        e_circle_ok = self._ring_structure_ok(patch)
      center_ok = self._center_ok(crop_bgr.shape[:2], best_loc, best_size)
      text_score = self._text_bonus(crop_bgr, best_loc, best_size)

    text_ok = (not self.require_text) or (text_score >= self.text_min_score)
    raw_hit = (
      best >= self.threshold
      and contrast_ok
      and center_ok
      and e_circle_ok
      and text_ok
    )
    if raw_hit:
      self._ok_streak += 1
    else:
      self._ok_streak = 0

    found = self._ok_streak >= self.confirm_frames
    return InteractionHit(
      found=found,
      score=self._last_score,
      scale=best_scale,
      loc=best_loc,
      contrast_ok=contrast_ok,
      center_ok=center_ok,
      e_circle_ok=e_circle_ok,
      text_score=text_score,
    )
