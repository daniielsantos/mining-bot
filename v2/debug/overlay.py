"""Overlay preview v2."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from v2.vendor.display import add_status_bar, fit_width
from v2.vendor.screen_ui import MiningScreenUI
from v2.core.types import FrameContext
from v2.navigation.bearing import forward_heading_deg


def _ray_end(px: float, py: float, angle_deg: float, length: float) -> tuple[int, int]:
  rad = math.radians(angle_deg)
  return int(round(px + length * math.cos(rad))), int(round(py + length * math.sin(rad)))


# Alinha com navigation.sticky_live_reacquire_px — rematch live do endpoint verde.
_LIVE_LOCK_MATCH_PX = 22.0


def _matched_live_blip(lock_x: float, lock_y: float, blips, *, tier: str | None = None):
  """Blip live mais perto do lock dentro do raio sticky (centro do disco)."""
  best = None
  best_d = _LIVE_LOCK_MATCH_PX
  for blip in blips:
    if tier is not None and blip.tier.lower() != tier.lower():
      continue
    d = math.hypot(blip.x - lock_x, blip.y - lock_y)
    if d <= best_d:
      best_d = d
      best = blip
  return best


def _ore_status_text(ctx: FrameContext) -> str | None:
  ore_score = ctx.meta.get("mining_ore_score")
  ore_found = bool(ctx.meta.get("mining_ore_found"))
  ore_thresh = ctx.meta.get("mining_ore_threshold")
  ore_near = bool(ctx.meta.get("mining_ore_near_miss"))
  if not isinstance(ore_score, (int, float)):
    return None
  thr = float(ore_thresh) if isinstance(ore_thresh, (int, float)) else 0.85
  if ore_found:
    return f"ore={ore_score:.2f} OK"
  if ore_near or ore_score >= thr - 0.08:
    return f"ore={ore_score:.2f}~{thr:.2f}"
  return f"ore={ore_score:.2f}<{thr:.2f}"


def _draw_hud_roi(
  frame: np.ndarray,
  roi: dict[str, Any],
  *,
  label: str | None = None,
  color: tuple[int, int, int] = (0, 255, 80),
  thickness: int = 2,
  marker: bool = True,
) -> np.ndarray:
  """Desenha retângulo da ROI Mining ore no frame de tela cheia."""
  out = frame.copy()
  _paint_hud_roi(
    out,
    roi,
    label=label,
    color=color,
    thickness=thickness,
    marker=marker,
  )
  return out


def _paint_hud_roi(
  frame: np.ndarray,
  roi: dict[str, Any],
  *,
  label: str | None = None,
  color: tuple[int, int, int] = (0, 255, 80),
  thickness: int = 2,
  marker: bool = True,
) -> None:
  """Desenha ROI in-place (sem copiar o frame)."""
  left = int(roi["left"])
  top = int(roi["top"])
  right = left + int(roi["width"]) - 1
  bottom = top + int(roi["height"]) - 1
  cv2.rectangle(frame, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)
  if marker:
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
  txt = label or f"ore ROI {roi['width']}x{roi['height']}"
  cv2.putText(
    frame,
    txt,
    (left, max(24, top - 8)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    color,
    2,
    cv2.LINE_AA,
  )

def _hud_overlay_panel(
  hud_vis: np.ndarray,
  panel_width: int,
  zoom: float,
) -> np.ndarray:
  """Zoom HUD strip for preview only (detection keeps the raw ROI).

  zoom=1 → same as fit_width (full ROI). zoom>1 → magnify then center-crop
  to panel_width so the Mining ore label is readable.
  """
  if hud_vis.size == 0:
    return np.zeros((36, panel_width, 3), dtype=np.uint8)
  zoom = max(0.25, float(zoom))
  h, w = hud_vis.shape[:2]
  scale = (panel_width / max(w, 1)) * zoom
  nw = max(1, int(round(w * scale)))
  nh = max(1, int(round(h * scale)))
  interp = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
  resized = cv2.resize(hud_vis, (nw, nh), interpolation=interp)
  if nw > panel_width:
    x0 = (nw - panel_width) // 2
    return resized[:, x0 : x0 + panel_width]
  if nw < panel_width:
    out = np.zeros((nh, panel_width, 3), dtype=resized.dtype)
    x0 = (panel_width - nw) // 2
    out[:, x0 : x0 + nw] = resized
    return out
  return resized


def render_overlay(
  ctx: FrameContext,
  *,
  screen_ui: MiningScreenUI,
  panel_width: int = 380,
  yellow_ray_px: float = 12.0,
  enabled: bool = False,
  game_focus: bool = False,
  calibrate_screen: np.ndarray | None = None,
  hud_roi: dict[str, Any] | None = None,
  draft_roi: dict[str, Any] | None = None,
  overlay_zoom: float = 2.5,
) -> np.ndarray:
  ore_score = ctx.meta.get("mining_ore_score")
  ore_status = _ore_status_text(ctx)
  first_person = bool(ctx.meta.get("first_person", True))
  fwd_deg = None
  live_match = None

  # Modo calibrar ROI: frame inteiro + retângulo salvo + rascunho (2 cliques).
  if calibrate_screen is not None and hud_roi is not None:
    dbg = calibrate_screen.copy()
    _paint_hud_roi(
      dbg,
      hud_roi,
      label=f"salvo {hud_roi['width']}x{hud_roi['height']}",
      color=(80, 180, 255),
      thickness=1,
      marker=False,
    )
    if draft_roi is not None:
      _paint_hud_roi(
        dbg,
        draft_roi,
        label=ore_status or f"novo {draft_roi['width']}x{draft_roi['height']}",
        color=(0, 255, 80),
        thickness=2,
        marker=True,
      )
    else:
      _paint_hud_roi(dbg, hud_roi, label=ore_status, color=(0, 255, 80))
    panel = fit_width(dbg, panel_width)
    step = (
      "2/2: clique no canto INFERIOR-DIREITO (ou solte o arraste)"
      if draft_roi is not None or ctx.meta.get("ore_roi_corner1")
      else "1/2: clique no canto SUPERIOR-ESQUERDO da barra escura"
    )
    lines = [
      f"CALIBRAR ore ROI | salvo={hud_roi['left']},{hud_roi['top']} "
      f"{hud_roi['width']}x{hud_roi['height']}"
      + (f" | {ore_status}" if ore_status else ""),
      step,
      f"ESC/Q sai | O cancela | ligado={enabled} foco={'sim' if game_focus else 'NAO'}",
    ]
    return add_status_bar(panel, lines, bar_height=68)

  dbg = ctx.minimap_bgr.copy()
  px, py = int(round(ctx.pivot[0])), int(round(ctx.pivot[1]))

  # Verde = caminho até o ALVO (nao e a frente da seta!).
  # Amarelo grosso = frente (1a pessoa: sempre cima na tela).
  arrow_state = ctx.arrow
  legacy = ctx.meta.get("legacy_arrow")
  fwd_deg = (
    forward_heading_deg(arrow_state, legacy, first_person=first_person)
    if arrow_state is not None
    else None
  )

  if ctx.lock is not None:
    live_match = _matched_live_blip(
      ctx.lock.x, ctx.lock.y, ctx.blips, tier=ctx.lock.tier
    )
  match_xy = (live_match.x, live_match.y) if live_match is not None else None
  for blip in ctx.blips:
    if match_xy is not None and math.hypot(blip.x - match_xy[0], blip.y - match_xy[1]) < 3:
      continue  # locked node: green line only (no ring)
    bx, by = int(round(blip.x)), int(round(blip.y))
    cv2.circle(dbg, (bx, by), 4, (180, 180, 180), 1)

  if ctx.lock is not None:
    # Verde = centro do blip live matched; senão ghost/lock congelado.
    if live_match is not None:
      tx, ty = int(round(live_match.x)), int(round(live_match.y))
    else:
      tx, ty = int(round(ctx.lock.x)), int(round(ctx.lock.y))
    cv2.line(dbg, (px, py), (tx, ty), (0, 180, 0), 1, cv2.LINE_AA)

  # Amarelo curto: so indica frente — NAO e distancia de chegada.
  # Comprimento: navigation.yellow_ray_px
  if fwd_deg is not None:
    ex, ey = _ray_end(px, py, fwd_deg, float(yellow_ray_px))
    cv2.line(dbg, (px, py), (ex, ey), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(dbg, (ex, ey), 2, (0, 255, 255), -1, cv2.LINE_AA)

  hud_result = ctx.meta.get("hud_result")
  if hud_result is not None:
    # Detect uses raw ctx.hud_bgr; zoom is display-only.
    hud_vis = screen_ui.debug_crop(ctx.hud_bgr, hud_result, status_text=ore_status)
    hud_panel = _hud_overlay_panel(hud_vis, panel_width, overlay_zoom)
    # Borda verde no strip zoomado (alinhamento visual da ROI).
    cv2.rectangle(
      hud_panel,
      (0, 0),
      (hud_panel.shape[1] - 1, hud_panel.shape[0] - 1),
      (0, 255, 80),
      1,
      cv2.LINE_AA,
    )
  else:
    hud_panel = np.zeros((36, panel_width, 3), dtype=np.uint8)
  if hud_panel.shape[0] < 36:
    hud_panel = cv2.resize(hud_panel, (panel_width, 36), interpolation=cv2.INTER_AREA)

  mini = fit_width(dbg, panel_width)
  panel = np.vstack([mini, hud_panel])

  brg = f"{ctx.bearing_deg:+.0f}" if ctx.bearing_deg is not None else "NA"
  side = ""
  if ctx.bearing_deg is not None:
    if abs(ctx.bearing_deg) <= 5:
      side = " ALINHADO"
    elif ctx.bearing_deg > 0:
      side = " →D"
    else:
      side = " ←E"
  lock_tag = f"id:{ctx.lock.track_id}" if ctx.lock else "sem-alvo"
  hold = ""
  if ctx.lock is not None and ctx.lock.pinned:
    hold += " PIN"
  if ctx.arrived:
    hold += " PAROU"
  sync = ""
  if ctx.lock is not None and live_match is not None:
    gap = math.hypot(live_match.x - ctx.lock.x, live_match.y - ctx.lock.y)
    if gap > 12:
      sync = f" desync={gap:.0f}px"
  nav = ctx.meta.get("nav_status", "")
  inv = " INV" if ctx.meta.get("turn_invert") else ""
  phase = ctx.meta.get("move_phase", "")
  phase_s = f" [{phase}]" if phase else ""
  dot = ctx.meta.get("target_dot")
  dot_s = f" dot={dot:+.0f}" if dot is not None else ""
  fwd_s = f" frente∠={fwd_deg:+.0f}°" if fwd_deg is not None else ""
  ore_s = (
    f" ore={ore_score:.2f}"
    if isinstance(ore_score, (int, float)) and ore_score > 0
    else ""
  )
  pulses = ctx.meta.get("final_pulses")
  pulse_s = f" pulses={pulses}" if isinstance(pulses, int) and pulses > 0 else ""
  if ctx.meta.get("arm_ore_roi"):
    arm = " | O/M: 2 cliques TL→BR na barra Mining ore"
  elif ctx.meta.get("arm_gray_pick"):
    arm = " | G: clique no no CINZA"
  else:
    arm = " | G=gray | O/M=ore ROI"
  lines = [
    f"{ctx.phase.value} | {lock_tag}{hold}{inv}{phase_s} | dist={ctx.dist_px:.0f}px brg={brg}{side}{dot_s}{fwd_s} | {ctx.action}{sync}",
    f"blips={len(ctx.blips)} alinhado={ctx.aligned}  {nav}{ore_s}{pulse_s}",
    (
      "1aPESSOA: amarelo=cima | verde=alvo | arrive→fine→close→probeE→READY→mine→SCAN | F8=proximo"
      if first_person
      else "AMARELO=nariz | verde=alvo | ligado={}".format(enabled)
    )
    + f" | ligado={enabled} foco={'sim' if game_focus else 'NAO'}{arm}",
  ]
  return add_status_bar(panel, lines, bar_height=68)
