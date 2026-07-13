"""
Calibra pixels_per_deg — camera em PRIMEIRA PESSOA.

Na 1a pessoa o MINIMAPA GIRA com a camera. Calibramos pelo angulo
do no cinza (nao so pela seta — o tracker pode nao ver a seta girar).

  cd mining_bot
  python -m v2.calibrate_camera

Requisitos:
  - 1a PESSOA (V), parado, no cinza visivel, GTA em foco

Globais (GTA em foco):
  Tab / CapsLock — passo manual +20 / -20 px
  F5/K      — calibracao | Home — teste

Janela: I inv, [ ] ppd, S salvar, Q sair
"""

from __future__ import annotations

import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT))

import v2.bootstrap as bootstrap

bootstrap.setup()

import cv2
import numpy as np
from pynput import keyboard

from v2.vendor.display import add_status_bar, fit_width
from v2.vendor.keyboard_input import (
  IS_WINDOWS,
  get_foreground_window_title,
  is_game_foreground,
  mouse_camera_look,
)
from v2.vendor.logger import mlog
from v2.vendor.navigator import normalize_angle_deg
from v2.capture.grabber import Grabber
from v2.core.config import load_config, save_overlay_patch
from v2.core.legacy import build_perception_stack
from v2.perception.pipeline import perceive

_FOCUS_ARM_S = 5.0
_SETTLE_S = 0.40
_MANUAL_STEP_PX = 20
_CAL_STEPS_PX = (30, 50, 70)
_MIN_ANGLE_DELTA = 0.8
_DEBUG_CAL_DIR = _ROOT / "debug_captures_cal"


@dataclass
class BlipPin:
  x: float
  y: float
  dist_px: float


@dataclass
class CalSession:
  active: bool = False
  armed: bool = False
  arm_deadline: float = 0.0
  step_idx: int = 0
  samples: list[float] = field(default_factory=list)
  logs: list[str] = field(default_factory=list)
  pin: BlipPin | None = None


def _read_facing(ctx) -> float | None:
  """Em 1a pessoa a seta nao gira — frente fixa = cima na tela (-90°)."""
  from v2.navigation.bearing import FIRST_PERSON_FORWARD_DEG

  if bool(ctx.meta.get("first_person", True)):
    return FIRST_PERSON_FORWARD_DEG
  if ctx.arrow.facing_deg is not None:
    return float(ctx.arrow.facing_deg)
  legacy = ctx.meta.get("legacy_arrow")
  if legacy is None:
    return None
  tip_x = getattr(legacy, "arrow_tip_x", None)
  tip_y = getattr(legacy, "arrow_tip_y", None)
  if tip_x is None or tip_y is None:
    return None
  px, py = ctx.pivot
  return math.degrees(math.atan2(float(tip_y) - py, float(tip_x) - px))


def _facing_delta(f0: float, f1: float) -> float:
  return normalize_angle_deg(f1 - f0)


def _minimap_diff(a: np.ndarray, b: np.ndarray) -> float:
  if a.shape != b.shape:
    return 999.0
  return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _screen_angle(ctx, xy: tuple[float, float]) -> float:
  px, py = ctx.pivot
  return math.degrees(math.atan2(xy[1] - py, xy[0] - px))


def _raw_tip(ctx) -> tuple[float, float] | None:
  legacy = ctx.meta.get("legacy_arrow")
  if legacy is None:
    return None
  tx = getattr(legacy, "arrow_tip_x", None)
  ty = getattr(legacy, "arrow_tip_y", None)
  if tx is None or ty is None:
    return None
  return float(tx), float(ty)


def _pin_from_blip(blip) -> BlipPin:
  return BlipPin(x=float(blip.x), y=float(blip.y), dist_px=float(blip.distance_px))


def _pick_blip(ctx, pin: BlipPin | None):
  if pin is None:
    return _pick_cal_blip(ctx, None)
  if not ctx.blips:
    return None
  return min(
    ctx.blips,
    key=lambda b: math.hypot(b.x - pin.x, b.y - pin.y) + abs(b.distance_px - pin.dist_px),
  )


def _match_blip_after(
  ctx,
  *,
  pin: BlipPin,
  a0: float,
  dx: int,
  bx0: float,
  by0: float,
):
  """Re-encontra o mesmo nó após girar — distância ao pivot quase constante."""
  best = None
  best_score = 999.0
  for b in ctx.blips:
    dist_err = abs(b.distance_px - pin.dist_px)
    if dist_err > 16:
      continue
    a1 = _screen_angle(ctx, (b.x, b.y))
    da = _facing_delta(a0, a1)
    if abs(da) > max(28.0, abs(dx) / 2.0):
      continue
    if dx > 0 and da >= 0:
      continue
    if dx < 0 and da <= 0:
      continue
    score = dist_err * 3.0 + math.hypot(b.x - bx0, b.y - by0) * 0.15
    if score < best_score:
      best_score = score
      best = b
  if best is not None:
    return best
  return _match_blip_by_distance(ctx, pin.dist_px)


def _match_blip_by_distance(ctx, ref_dist_px: float):
  if not ctx.blips:
    return None
  return min(ctx.blips, key=lambda b: abs(b.distance_px - ref_dist_px))


def _pick_cal_blip(ctx, pinned: tuple[float, float] | None) -> Any | None:
  if pinned is not None:
    if not ctx.blips:
      return None
    return min(ctx.blips, key=lambda b: math.hypot(b.x - pinned[0], b.y - pinned[1]))
  ranked = sorted(ctx.blips, key=lambda b: b.distance_px)
  for blip in ranked:
    if blip.distance_px >= 15:
      return blip
  return ranked[0] if ranked else None


def _rotation_ok(dx: int, delta: float, metric: str) -> bool:
  """Valida se o giro medido bate com o mouse enviado."""
  if metric == "nó":
    # Camera direita (+dx) -> angulo do no na tela cai (delta negativo).
    return (dx > 0 and delta < 0) or (dx < 0 and delta > 0)
  return (dx > 0 and delta > 0) or (dx < 0 and delta < 0)


def _pulse_result(
  grabber: Grabber,
  *,
  arrow_tracker: Any,
  node_detector: Any,
  screen_ui: Any,
  cfg: dict[str, Any],
  dx: int,
  hold_rmb: bool,
  look_invert: bool,
  mouse_backend: str,
  pin: BlipPin | None,
) -> tuple[float | None, str]:
  """Move mouse e mede graus pelo angulo do no fixo no minimapa."""
  del look_invert
      minimap0, hud0 = grabber.grab()
  ctx0 = perceive(0, minimap0, hud0, arrow_tracker=arrow_tracker, node_detector=node_detector, screen_ui=screen_ui, cfg=cfg)
  blip0 = _pick_blip(ctx0, pin)
  if blip0 is None:
    return None, "precisa nó cinza no minimapa — pare perto de um"

  pulse_pin = pin or _pin_from_blip(blip0)
  a0 = _screen_angle(ctx0, (blip0.x, blip0.y))
  bx0, by0 = blip0.x, blip0.y

  used = mouse_camera_look(dx, 0, hold_rmb=hold_rmb, backend=mouse_backend)
  time.sleep(_SETTLE_S)
      minimap1, hud1 = grabber.grab()
  ctx1 = perceive(0, minimap1, hud1, arrow_tracker=arrow_tracker, node_detector=node_detector, screen_ui=screen_ui, cfg=cfg)
  diff = _minimap_diff(minimap0, minimap1)

  blip1 = _match_blip_after(ctx1, pin=pulse_pin, a0=a0, dx=dx, bx0=bx0, by0=by0)
  if blip1 is None:
    return None, f"dx={dx:+d} perdeu nó após pulso (diff={diff:.1f})"

  a1 = _screen_angle(ctx1, (blip1.x, blip1.y))
  da = _facing_delta(a0, a1)
  delta = da
  metric = "nó"

  if abs(delta) < _MIN_ANGLE_DELTA:
    _DEBUG_CAL_DIR.mkdir(parents=True, exist_ok=True)
    n = len(list(_DEBUG_CAL_DIR.glob("*"))) // 2 + 1
    cv2.imwrite(str(_DEBUG_CAL_DIR / f"{n:03d}_before.jpg"), minimap0)
    cv2.imwrite(str(_DEBUG_CAL_DIR / f"{n:03d}_after.jpg"), minimap1)
    return None, (
      f"dx={dx:+d} [{used}] diff={diff:.1f} | nó {a0:+.1f}°->{a1:+.1f}° (Δ{da:+.1f}°) — sem giro"
    )

  if abs(da) > max(28.0, abs(dx) / 2.0):
    return None, (
      f"dx={dx:+d} Δnó={da:+.1f}° grande — trocou nó? F5 de novo com 1 nó visível ou P fixar"
    )

  if not _rotation_ok(dx, delta, metric):
    return None, f"dx={dx:+d} Δnó={delta:+.1f}° invertido — pressione I"

  ppd = abs(dx) / abs(delta)
  return (
    ppd,
    f"dx={dx:+d} [{used}] | nó Δ{delta:+.1f}° | {a0:+.1f}°->{a1:+.1f}° | "
    f"ppd={ppd:.2f} | diff={diff:.1f}",
  )


def _finish_cal(session: CalSession) -> tuple[float | None, str]:
  session.active = False
  session.step_idx = 0
  if len(session.samples) >= 2:
    med = float(statistics.median(session.samples))
    return med, f"CAL OK ppd={med:.2f} (n={len(session.samples)}) | {session.logs[-1]}"
  if len(session.samples) == 1:
    return session.samples[0], f"CAL parcial ppd={session.samples[0]:.2f} | {session.logs[0]}"
  tail = session.logs[-1] if session.logs else "sem amostras"
  return None, f"CAL falhou: {tail}"


def _cal_tick(
  session: CalSession,
  grabber: Grabber,
  *,
  arrow_tracker: Any,
  node_detector: Any,
  screen_ui: Any,
  cfg: dict[str, Any],
  look_invert: bool,
  hold_rmb: bool,
  mouse_backend: str,
  pin: BlipPin | None,
) -> tuple[tuple[float | None, str] | None, str | None]:
  if not session.active or session.step_idx >= len(_CAL_STEPS_PX):
    return None, None

  dx = _CAL_STEPS_PX[session.step_idx]
  signed = dx if not look_invert else -dx
  ppd, msg = _pulse_result(
    grabber,
    arrow_tracker=arrow_tracker,
    node_detector=node_detector,
    screen_ui=screen_ui,
    cfg=cfg,
    dx=signed,
    hold_rmb=hold_rmb,
    look_invert=look_invert,
    mouse_backend=mouse_backend,
    pin=pin,
  )
  session.logs.append(msg)
  if ppd is not None:
    session.samples.append(ppd)
  session.step_idx += 1
  if session.step_idx >= len(_CAL_STEPS_PX):
    return _finish_cal(session), msg
  return None, f"pulso {session.step_idx}/{len(_CAL_STEPS_PX)}: {msg}"


def _draw_panel(
  ctx,
  *,
  facing: float | None,
  bearing: float | None,
  pixels_per_deg: float,
  look_invert: bool,
  status: str,
  panel_width: int,
  tick: int,
  target_xy: tuple[float, float] | None,
) -> np.ndarray:
  dbg = ctx.minimap_bgr.copy()
  px, py = int(ctx.pivot[0]), int(ctx.pivot[1])
  cv2.drawMarker(dbg, (px, py), (0, 80, 255), cv2.MARKER_CROSS, 14, 2)

  legacy = ctx.meta.get("legacy_arrow")
  if legacy is not None and legacy.arrow_tip_x is not None and legacy.arrow_tip_y is not None:
    cv2.line(
      dbg,
      (px, py),
      (int(legacy.arrow_tip_x), int(legacy.arrow_tip_y)),
      (0, 220, 255),
      2,
    )

  for blip in ctx.blips:
    cv2.circle(dbg, (int(blip.x), int(blip.y)), 4, (160, 160, 160), 1)

  if target_xy is not None:
    tx, ty = int(target_xy[0]), int(target_xy[1])
    cv2.line(dbg, (px, py), (tx, ty), (0, 255, 0), 1)
    cv2.circle(dbg, (tx, ty), 6, (255, 220, 0), 1)

  face_s = f"{facing:+.1f}" if facing is not None else "NA"
  brg_s = f"{bearing:+.0f}" if bearing is not None else "NA"
  inv = " SIM" if look_invert else " nao"
  lines = [
    f"#{tick} 1aPESSOA | facing={face_s}° | ang_nó={brg_s}° | ppd={pixels_per_deg:.2f} inv={inv}",
    status,
    "Tab/CapsLock=±20px F5/K=cal Home=teste | I inv H=rmb B=backend S=salvar",
  ]
  return add_status_bar(fit_width(dbg, panel_width), lines, bar_height=68)


def _arm_cal(session: CalSession, now: float, focused: bool) -> str:
  session.armed = True
  session.arm_deadline = now + _FOCUS_ARM_S
  if focused:
    return "F5/K — fixando nó..."
  return f"F5/K — clique no GTA ({_FOCUS_ARM_S:.0f}s max, 1a PESSOA)"


def _start_cal(session: CalSession, ctx) -> str:
  blip = _pick_cal_blip(ctx, None)
  if blip is None:
    session.armed = False
    return "sem nó cinza no minimapa — pare perto de um"
  session.pin = _pin_from_blip(blip)
  session.armed = False
  session.active = True
  session.step_idx = 0
  session.samples.clear()
  session.logs.clear()
  return f"CAL nó fixado dist={blip.distance_px:.0f}px — pulso 1/3"


def main() -> None:
  cfg = load_config()
  nav = cfg.setdefault("navigation", {})
  cam = nav.setdefault("camera", {})
  pixels_per_deg = float(cam.get("pixels_per_deg", 7.5))
  look_invert = bool(cam.get("look_invert", False))
  hold_rmb = bool(cam.get("hold_rmb_for_look", False))
  mouse_backend = str(cam.get("mouse_backend", "auto"))
  panel_width = int(cfg.get("debug", {}).get("preview_panel_width", 380))
  focus_kw = cfg.get("game_foreground_keywords")
  focus_proc = cfg.get("game_process_names")

  node_detector, arrow_tracker, screen_ui = build_perception_stack(cfg)
  status = "1a PESSOA + nó cinza | Tab/CapsLock passo manual (GTA em foco)"
  last_msg = ""
  tick = 0
  cal = CalSession()
  test_armed = False
  test_deadline = 0.0
  manual_pin: BlipPin | None = None

  def active_pin() -> BlipPin | None:
    return cal.pin if cal.active else manual_pin

  request_lock = threading.Lock()
  request_cal = False
  request_test = False
  request_manual_px = 0

  def game_focus() -> bool:
    return (not IS_WINDOWS) or is_game_foreground(keywords=focus_kw, process_names=focus_proc)

  def on_global_press(key: keyboard.Key | keyboard.KeyCode) -> None:
    nonlocal request_cal, request_test, request_manual_px
    with request_lock:
      if key in (keyboard.Key.f5, keyboard.Key.f10):
        request_cal = True
      elif key == keyboard.Key.home:
        request_test = True
      elif key == keyboard.Key.caps_lock:
        request_manual_px = -_MANUAL_STEP_PX
      elif key == keyboard.Key.tab:
        request_manual_px = _MANUAL_STEP_PX

  keyboard.Listener(on_press=on_global_press, daemon=True).start()

  cv2.namedWindow("Camera Cal v2", cv2.WINDOW_AUTOSIZE)
  print(__doc__)
  mlog("[cam-cal] Tab/CapsLock=±20px | F5/K=calibrar (3 pulsos → ppd) | Home=teste")

  with Grabber(cfg) as grabber:
    while True:
      tick += 1
      minimap, hud = grabber.grab()
      ctx = perceive(
        tick,
        minimap,
        hud,
        arrow_tracker=arrow_tracker,
        node_detector=node_detector,
        screen_ui=screen_ui,
        cfg=cfg,
      )

      with request_lock:
        do_cal = request_cal
        do_test = request_test
        manual_px = request_manual_px
        request_cal = False
        request_test = False
        request_manual_px = 0

      blip = _pick_blip(ctx, active_pin())
      target_xy = (blip.x, blip.y) if blip else None
      facing = _read_facing(ctx)
      bearing = _screen_angle(ctx, target_xy) if target_xy else None
      focused = game_focus()
      now = time.perf_counter()

      if do_cal and not cal.active:
        if focused:
          last_msg = _start_cal(cal, ctx)
        else:
          last_msg = _arm_cal(cal, now, focused=False)

      if cal.armed and not cal.active:
        if focused:
          last_msg = _start_cal(cal, ctx)
        elif now > cal.arm_deadline:
          cal.armed = False
          last_msg = f"timeout ({get_foreground_window_title()!r}) — F5/K de novo"

      if cal.active:
        # Um pulso por frame de UI (cada pulso ~0.4s dentro de _pulse_result).
        if cal.step_idx < len(_CAL_STEPS_PX):
          result, step_msg = _cal_tick(
            cal,
            grabber,
            arrow_tracker=arrow_tracker,
            node_detector=node_detector,
            screen_ui=screen_ui,
            cfg=cfg,
            look_invert=look_invert,
            hold_rmb=hold_rmb,
            mouse_backend=mouse_backend,
            pin=cal.pin,
          )
          if step_msg:
            last_msg = step_msg
          if result is not None:
            ppd, msg = result
            last_msg = msg
            cal.active = False
            if ppd is not None:
              pixels_per_deg = ppd
            mlog(f"[cam-cal] {msg}")

      if manual_px != 0 and focused and not cal.active:
        signed = manual_px if not look_invert else -manual_px
        _, last_msg = _pulse_result(
          grabber,
          arrow_tracker=arrow_tracker,
          node_detector=node_detector,
          screen_ui=screen_ui,
          cfg=cfg,
          dx=signed,
          hold_rmb=hold_rmb,
          look_invert=look_invert,
          mouse_backend=mouse_backend,
          pin=active_pin(),
        )
        mlog(f"[cam-cal] manual {last_msg}")

      if do_test and not cal.active:
        if focused:
          ppd, msg = _pulse_result(
            grabber,
            arrow_tracker=arrow_tracker,
            node_detector=node_detector,
            screen_ui=screen_ui,
            cfg=cfg,
            dx=40 if not look_invert else -40,
            hold_rmb=hold_rmb,
            look_invert=look_invert,
            mouse_backend=mouse_backend,
            pin=active_pin(),
          )
          last_msg = msg
          mlog(f"[cam-cal] teste {msg}")
        else:
          test_armed = True
          test_deadline = now + _FOCUS_ARM_S
          last_msg = f"Home — clique no GTA ({_FOCUS_ARM_S:.0f}s)..."

      if test_armed and not cal.active:
        if focused:
          test_armed = False
          _, last_msg = _pulse_result(
            grabber,
            arrow_tracker=arrow_tracker,
            node_detector=node_detector,
            screen_ui=screen_ui,
            cfg=cfg,
            dx=40 if not look_invert else -40,
            hold_rmb=hold_rmb,
            look_invert=look_invert,
            mouse_backend=mouse_backend,
            pin=active_pin(),
          )
          mlog(f"[cam-cal] teste {last_msg}")
        elif now > test_deadline:
          test_armed = False
          last_msg = "timeout teste — Home de novo"

      hint = ""
      if cal.armed or test_armed:
        hint = f" | aguardando GTA {max(0.0, max(cal.arm_deadline, test_deadline) - now):.1f}s"
      elif not focused:
        hint = " | sem foco — OK preview; calibrar com GTA ativo"

      panel = _draw_panel(
        ctx,
        facing=facing,
        bearing=bearing,
        pixels_per_deg=pixels_per_deg,
        look_invert=look_invert,
        status=(last_msg or status) + hint,
        panel_width=panel_width,
        tick=tick,
        target_xy=target_xy,
      )
      cv2.imshow("Camera Cal v2", panel)
      key = cv2.waitKey(1) & 0xFF

      if key in (ord("q"), 27):
        break
      if key == ord("k") and not cal.active:
        if focused:
          last_msg = _start_cal(cal, ctx)
        else:
          last_msg = _arm_cal(cal, now, focused=False)
      elif key == ord("i"):
        look_invert = not look_invert
        last_msg = f"look_invert={look_invert}"
      elif key == ord("["):
        pixels_per_deg = max(1.0, pixels_per_deg - 0.5)
        last_msg = f"ppd={pixels_per_deg:.2f}"
      elif key == ord("]"):
        pixels_per_deg = min(50.0, pixels_per_deg + 0.5)
        last_msg = f"ppd={pixels_per_deg:.2f}"
      elif key == ord("h"):
        hold_rmb = not hold_rmb
        last_msg = f"hold_rmb={hold_rmb}"
      elif key == ord("b"):
        mouse_backend = {"auto": "both", "both": "mouse_event", "mouse_event": "sendinput"}.get(
          mouse_backend, "auto"
        )
        last_msg = f"mouse_backend={mouse_backend}"
      elif key == ord("p") and blip is not None:
        manual_pin = _pin_from_blip(blip)
        last_msg = f"nó fixado dist={blip.distance_px:.0f}px"
      elif key == ord("s"):
        path = save_overlay_patch(
          {
            "navigation": {
              "control_mode": "camera",
              "camera": {
                "pixels_per_deg": round(pixels_per_deg, 2),
                "look_invert": look_invert,
                "hold_rmb_for_look": hold_rmb,
                "mouse_backend": mouse_backend,
                "first_person": True,
              },
            },
          }
        )
        last_msg = f"salvo {path.name} ppd={pixels_per_deg:.2f}"
        mlog(f"[cam-cal] {last_msg}")
      elif key in (ord(","), ord(".")):
        last_msg = "use Tab/CapsLock com GTA em foco ([ ] ajusta ppd na janela)"

  cv2.destroyAllWindows()
  mlog("[cam-cal] Encerrado")


if __name__ == "__main__":
  main()
