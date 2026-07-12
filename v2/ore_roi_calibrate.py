"""Calibra ROI da barra Mining ore e grava em v2/config.json.

Uso no preview (python -m v2.main --preview):
  O ou M — abre janela grande "Calibrar Mining ore ROI"
  1º clique = canto superior-esquerdo da barra escura "Mining ore"
  2º clique = canto inferior-direito
  → grava navigation.mining_ore.roi (+ screen_roi), aplica ao vivo

Também dá para arrastar: pressiona no canto TL, solta no BR.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from v2.core.config import get_hud_roi, save_overlay_patch

# Faixa tipica da barra escura (label + progress) em 1440p — nao um strip largo.
DEFAULT_ORE_ROI_WIDTH = 520
DEFAULT_ORE_ROI_HEIGHT = 56
MIN_ORE_ROI_WIDTH = 80
MIN_ORE_ROI_HEIGHT = 24

ORE_CAL_WINDOW_NAME = "Calibrar Mining ore ROI"
DEFAULT_CAL_WINDOW_W = 1600
DEFAULT_CAL_WINDOW_H = 900


def clamp(value: int, lo: int, hi: int) -> int:
  return max(lo, min(hi, value))


def screen_size(cfg: dict[str, Any]) -> tuple[int, int]:
  res = cfg.get("resolution", {})
  return int(res.get("width", 2560)), int(res.get("height", 1440))


def roi_from_corners(
  x1: int,
  y1: int,
  x2: int,
  y2: int,
  *,
  screen_w: int,
  screen_h: int,
  min_width: int = MIN_ORE_ROI_WIDTH,
  min_height: int = MIN_ORE_ROI_HEIGHT,
) -> dict[str, int]:
  """ROI a partir de dois cantos (qualquer ordem), limitado à tela."""
  left = clamp(min(int(x1), int(x2)), 0, max(0, screen_w - 1))
  right = clamp(max(int(x1), int(x2)), 0, max(0, screen_w - 1))
  top = clamp(min(int(y1), int(y2)), 0, max(0, screen_h - 1))
  bottom = clamp(max(int(y1), int(y2)), 0, max(0, screen_h - 1))
  width = max(int(min_width), right - left + 1)
  height = max(int(min_height), bottom - top + 1)
  if left + width > screen_w:
    left = max(0, screen_w - width)
    width = min(width, screen_w)
  if top + height > screen_h:
    top = max(0, screen_h - height)
    height = min(height, screen_h)
  return {"left": left, "top": top, "width": width, "height": height}


def roi_centered_on(
  cx: int,
  cy: int,
  *,
  width: int,
  height: int,
  screen_w: int,
  screen_h: int,
) -> dict[str, int]:
  """ROI com centro em (cx, cy), limitado à tela."""
  w = max(MIN_ORE_ROI_WIDTH, int(width))
  h = max(MIN_ORE_ROI_HEIGHT, int(height))
  left = clamp(int(round(cx - w / 2.0)), 0, max(0, screen_w - w))
  top = clamp(int(round(cy - h / 2.0)), 0, max(0, screen_h - h))
  if left + w > screen_w:
    w = max(MIN_ORE_ROI_WIDTH, screen_w - left)
  if top + h > screen_h:
    h = max(MIN_ORE_ROI_HEIGHT, screen_h - top)
  return {"left": left, "top": top, "width": w, "height": h}


def mining_ore_roi_patch(roi: dict[str, int]) -> dict[str, Any]:
  """Patch: navigation.mining_ore.roi + screen_roi legado (mesma faixa)."""
  box = {
    "left": int(roi["left"]),
    "top": int(roi["top"]),
    "width": int(roi["width"]),
    "height": int(roi["height"]),
  }
  return {
    "screen_roi": dict(box),
    "navigation": {"mining_ore": {"roi": dict(box)}},
  }


def save_mining_ore_roi(roi: dict[str, int]):
  """Persiste ROI em mining_bot/v2/config.json."""
  return save_overlay_patch(mining_ore_roi_patch(roi))


def apply_mining_ore_roi(cfg: dict[str, Any], roi: dict[str, int]) -> dict[str, int]:
  """Atualiza cfg em memória (Grabber lê get_hud_roi a cada frame)."""
  box = {
    "left": int(roi["left"]),
    "top": int(roi["top"]),
    "width": int(roi["width"]),
    "height": int(roi["height"]),
  }
  cfg["screen_roi"] = dict(box)
  nav = cfg.setdefault("navigation", {})
  ore = nav.setdefault("mining_ore", {})
  ore["roi"] = dict(box)
  return box


def place_roi_at_click(
  cfg: dict[str, Any],
  screen_x: int,
  screen_y: int,
) -> dict[str, int]:
  """Centra ROI atual (ou default fino) no clique em coords de tela."""
  current = get_hud_roi(cfg)
  sw, sh = screen_size(cfg)
  return roi_centered_on(
    screen_x,
    screen_y,
    width=int(current.get("width", DEFAULT_ORE_ROI_WIDTH)),
    height=int(current.get("height", DEFAULT_ORE_ROI_HEIGHT)),
    screen_w=sw,
    screen_h=sh,
  )


def place_roi_from_clicks(
  cfg: dict[str, Any],
  x1: int,
  y1: int,
  x2: int,
  y2: int,
) -> dict[str, int]:
  """Dois cliques (TL/BR ou qualquer ordem) → ROI limitada à tela."""
  sw, sh = screen_size(cfg)
  return roi_from_corners(x1, y1, x2, y2, screen_w=sw, screen_h=sh)


def map_preview_click_to_screen(
  click_x: int,
  click_y: int,
  *,
  frame_hw: tuple[int, int],
  display_width: int,
  display_height: int | None = None,
) -> tuple[int, int] | None:
  """Clique na imagem calibrada (escala uniforme) → pixel de tela absoluto.

  Se display_height for None, usa a mesma regra que fit_width
  (scale = display_width / frame_w).
  """
  fh, fw = int(frame_hw[0]), int(frame_hw[1])
  if fw <= 0 or fh <= 0 or display_width <= 0:
    return None
  if display_height is None:
    scale = display_width / float(fw)
    disp_h = max(1, int(fh * scale))
    disp_w = int(display_width)
  else:
    disp_w = int(display_width)
    disp_h = int(display_height)
    if disp_w <= 0 or disp_h <= 0:
      return None
    scale = min(disp_w / float(fw), disp_h / float(fh))
    disp_w = max(1, int(round(fw * scale)))
    disp_h = max(1, int(round(fh * scale)))
  if click_y < 0 or click_y >= disp_h or click_x < 0 or click_x >= disp_w:
    return None
  sx = clamp(int(round(click_x / scale)), 0, fw - 1)
  sy = clamp(int(round(click_y / scale)), 0, fh - 1)
  return sx, sy


def monitor_work_area() -> tuple[int, int]:
  """Área útil do monitor (margem para barra de título / taskbar)."""
  try:
    import ctypes

    user32 = ctypes.windll.user32
    sw = int(user32.GetSystemMetrics(0))
    sh = int(user32.GetSystemMetrics(1))
    return max(800, sw - 80), max(600, sh - 120)
  except Exception:
    return DEFAULT_CAL_WINDOW_W, DEFAULT_CAL_WINDOW_H


def calibrate_display_size(
  frame_w: int,
  frame_h: int,
  *,
  max_w: int | None = None,
  max_h: int | None = None,
) -> tuple[int, int, float]:
  """Escala o frame do jogo para caber no monitor (preferência ~1600×900+)."""
  mw, mh = monitor_work_area()
  if max_w is not None:
    mw = int(max_w)
  if max_h is not None:
    mh = int(max_h)
  # Quase fullscreen no monitor; floor mínimo útil se o monitor for pequeno.
  mw = max(mw, min(DEFAULT_CAL_WINDOW_W, mw))
  mh = max(mh, min(DEFAULT_CAL_WINDOW_H, mh))
  fw = max(1, int(frame_w))
  fh = max(1, int(frame_h))
  scale = min(mw / float(fw), mh / float(fh))
  dw = max(1, int(round(fw * scale)))
  dh = max(1, int(round(fh * scale)))
  return dw, dh, float(scale)


def open_ore_calibrate_window(frame_w: int, frame_h: int) -> tuple[int, int]:
  """Abre janela grande WINDOW_NORMAL e retorna (disp_w, disp_h)."""
  dw, dh, _ = calibrate_display_size(frame_w, frame_h)
  cv2.namedWindow(ORE_CAL_WINDOW_NAME, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(ORE_CAL_WINDOW_NAME, dw, dh)
  return dw, dh


def close_ore_calibrate_window() -> None:
  try:
    cv2.destroyWindow(ORE_CAL_WINDOW_NAME)
  except Exception:
    pass


def _paint_roi_box(
  frame: np.ndarray,
  roi: dict[str, Any],
  *,
  label: str,
  color: tuple[int, int, int],
  thickness: int,
  marker: bool,
) -> None:
  left = int(roi["left"])
  top = int(roi["top"])
  right = left + int(roi["width"]) - 1
  bottom = top + int(roi["height"]) - 1
  cv2.rectangle(frame, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)
  if marker:
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 28, max(2, thickness), cv2.LINE_AA)
  cv2.putText(
    frame,
    label,
    (left, max(36, top - 12)),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    color,
    2,
    cv2.LINE_AA,
  )


def render_ore_calibrate_view(
  screen: np.ndarray,
  hud_roi: dict[str, Any],
  *,
  draft_roi: dict[str, Any] | None = None,
  has_corner1: bool = False,
  display_wh: tuple[int, int] | None = None,
  status_line: str | None = None,
) -> np.ndarray:
  """Frame de jogo + retângulos grandes, escalado para a janela de calibração."""
  dbg = screen.copy()
  _paint_roi_box(
    dbg,
    hud_roi,
    label=f"salvo {hud_roi['width']}x{hud_roi['height']}",
    color=(80, 180, 255),
    thickness=2,
    marker=False,
  )
  if draft_roi is not None:
    _paint_roi_box(
      dbg,
      draft_roi,
      label=f"novo {draft_roi['width']}x{draft_roi['height']}",
      color=(0, 255, 80),
      thickness=4,
      marker=True,
    )
  else:
    _paint_roi_box(
      dbg,
      hud_roi,
      label="ROI atual (verde)",
      color=(0, 255, 80),
      thickness=4,
      marker=True,
    )

  fh, fw = dbg.shape[:2]
  if display_wh is None:
    dw, dh, _ = calibrate_display_size(fw, fh)
  else:
    dw, dh = int(display_wh[0]), int(display_wh[1])
  scale = min(dw / float(fw), dh / float(fh))
  out_w = max(1, int(round(fw * scale)))
  out_h = max(1, int(round(fh * scale)))
  interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
  panel = cv2.resize(dbg, (out_w, out_h), interpolation=interp)

  # Instruções no próprio frame (sem barra extra) para o clique mapear 1:1.
  step = (
    "2/2: clique no canto INFERIOR-DIREITO (ou solte o arraste)"
    if draft_roi is not None or has_corner1
    else "1/2: clique no canto SUPERIOR-ESQUERDO da barra escura Mining ore"
  )
  lines = [
    f"CALIBRAR Mining ore ROI | salvo={hud_roi['left']},{hud_roi['top']} "
    f"{hud_roi['width']}x{hud_roi['height']}",
    step,
    "ESC/Q cancela | O cancela | 2 cliques TL→BR (ou arraste)",
  ]
  if status_line:
    lines.append(status_line)
  pad = 10
  line_h = 28
  box_h = pad * 2 + line_h * len(lines)
  cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, box_h), (0, 0, 0), -1)
  for i, line in enumerate(lines):
    cv2.putText(
      panel,
      line,
      (14, pad + 20 + i * line_h),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.72,
      (0, 255, 80),
      2,
      cv2.LINE_AA,
    )
  return panel
