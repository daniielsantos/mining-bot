"""
Navegação por câmera (1a pessoa).

- Sem A/D: correção só com pulsos de mouse (câmera).
- W quando o alvo está à frente o bastante.
- align_only: só gira câmera (sem W) — útil p/ calibrar.
- fine_align: parado no nó — pulsos pequenos até ponta amarela no centro.
- close_walk: pós-align — só W até close_walk_px de progresso no minimapa (sem câmera).
- final_approach: probe E inicial; miss → W pulse + wait + E (até final_pulse_max); keep E se Mining ore.
"""

from __future__ import annotations

import time
from typing import Any

from keyboard_input import IS_WINDOWS, mouse_camera_look, press_key, release_all_keys, release_key, tap_key


class CameraAlignController:
  def __init__(self, cfg: dict[str, Any]) -> None:
    nav = cfg.get("navigation", {})
    cam = nav.get("camera", {})
    self.align_only = bool(cam.get("align_only", False))
    self.align_deg = float(nav.get("align_deg", 10))
    # Acima disso: para W e só realinha câmera.
    self.walk_max_deg = float(cam.get("walk_max_deg", 12))
    self.pixels_per_deg = float(cam.get("pixels_per_deg", 7.5))
    self.max_step_px = int(cam.get("max_step_px", 20))
    self.min_step_px = int(cam.get("min_step_px", 2))
    # Pulsos menores enquanto anda (evita zig-zag).
    self.walk_max_step_px = int(cam.get("walk_max_step_px", 10))
    self.look_deadband_deg = float(cam.get("look_deadband_deg", 3.5))
    self.look_interval_ms = float(cam.get("look_interval_ms", 130))
    self.look_invert = bool(cam.get("look_invert", False))
    self.hold_rmb = bool(cam.get("hold_rmb_for_look", False))
    self.mouse_backend = str(cam.get("mouse_backend", "auto"))
    # Fine-align pós-arrive: pulsos menores que walk/realign.
    self.fine_align_deadband_deg = float(cam.get("fine_align_deadband_deg", 1.5))
    self.fine_align_pixels_per_deg = float(
      cam.get(
        "fine_align_pixels_per_deg",
        max(2.0, self.pixels_per_deg * 0.35),
      )
    )
    self.fine_align_max_step_px = int(cam.get("fine_align_max_step_px", 4))
    self.fine_align_min_step_px = int(cam.get("fine_align_min_step_px", 1))
    self.fine_align_look_interval_ms = float(
      cam.get("fine_align_look_interval_ms", 90)
    )
    # Pós close-walk: pulsos W + probe E até Mining ore.
    self.final_pulse_w_ms = float(nav.get("final_pulse_w_ms", 350.0))
    self.final_pulse_interval_ms = float(nav.get("final_pulse_interval_ms", 280.0))
    self.probe_e_ms = float(nav.get("final_probe_e_ms", 750.0))
    self.interact_hold_ms = float(cfg.get("walker", {}).get("interact_hold_ms", 80))
    # True quando Mining ore apareceu mid-hold — não soltar E.
    self.e_held_for_mine = False
    self._last_action = "idle"
    self._last_look_at = 0.0
    self._last_pulse_at = 0.0

  @property
  def last_action(self) -> str:
    return self._last_action

  def stop(self) -> str:
    if IS_WINDOWS:
      if self.e_held_for_mine:
        # Mantém E; solta só movimento.
        for key in ("w", "a", "d", "s"):
          release_key(key)
      else:
        release_all_keys()
    self._last_action = "idle"
    return self._last_action

  def _steer_mouse(
    self,
    bearing_deg: float,
    *,
    max_step: int | None = None,
    deadband_deg: float | None = None,
    pixels_per_deg: float | None = None,
    min_step: int | None = None,
  ) -> int:
    dead = self.look_deadband_deg if deadband_deg is None else deadband_deg
    if abs(bearing_deg) < dead:
      return 0
    cap = self.max_step_px if max_step is None else max_step
    scale = self.pixels_per_deg if pixels_per_deg is None else pixels_per_deg
    floor = self.min_step_px if min_step is None else min_step
    raw = bearing_deg * scale
    if raw > 0:
      dx = max(floor, min(cap, int(raw)))
    elif raw < 0:
      dx = -max(floor, min(cap, int(-raw)))
    else:
      dx = 0
    if self.look_invert:
      dx = -dx
    return dx

  def _apply_look(
    self,
    dx: int,
    dy: int = 0,
    *,
    interval_ms: float | None = None,
  ) -> bool:
    if dx == 0 and dy == 0:
      return False
    now = time.perf_counter()
    gap = self.look_interval_ms if interval_ms is None else interval_ms
    if (now - self._last_look_at) * 1000.0 < gap:
      return False
    mouse_camera_look(
      dx, dy, hold_rmb=self.hold_rmb, backend=self.mouse_backend
    )
    self._last_look_at = now
    return True

  def look_yaw_deg(
    self,
    yaw_deg: float,
    *,
    interval_ms: float | None = None,
  ) -> str:
    """
    Pulse de yaw da câmera (sem W).

    yaw_deg negativo = olhar/girar à esquerda (antes de look_invert),
    igual ao sinal de bearing em `_steer_mouse`.
    pixels = yaw_deg * pixels_per_deg (sem cap de walk max_step).
    """
    if not IS_WINDOWS:
      self._last_action = "spin-skip"
      return self._last_action
    raw = float(yaw_deg) * self.pixels_per_deg
    if raw > 0:
      dx = max(1, int(round(raw)))
    elif raw < 0:
      dx = -max(1, int(round(-raw)))
    else:
      self._last_action = "spin-0"
      return self._last_action
    if self.look_invert:
      dx = -dx
    if self._apply_look(dx, interval_ms=interval_ms):
      self._last_action = f"spin-look-{dx:+d}"
    else:
      self._last_action = "spin-wait"
    return self._last_action

  def pulse_forward(self, *, hold_ms: float | None = None) -> str:
    """Tap curto em W (nao segura continuo)."""
    if not IS_WINDOWS:
      self._last_action = "pulse-skip"
      return self._last_action
    release_key("w")
    ms = self.final_pulse_w_ms if hold_ms is None else float(hold_ms)
    tap_key("w", hold_ms=max(1.0, ms))
    self._last_pulse_at = time.perf_counter()
    self._last_action = f"pulse-w-{ms:.0f}ms"
    return self._last_action

  def pulse_space(self, *, hold_ms: float = 150.0) -> str:
    """STUCK recovery step A: tap Space (jump) before D strafe."""
    if not IS_WINDOWS:
      self._last_action = "space-skip"
      return self._last_action
    release_key("w")
    release_key("a")
    release_key("d")
    ms = max(1.0, float(hold_ms))
    tap_key("space", hold_ms=ms)
    self._last_action = f"space-{ms:.0f}ms"
    return self._last_action

  def pulse_strafe_d(self, *, hold_ms: float = 2000.0) -> str:
    """STUCK recovery: solta W, segura D, solta D (sem A/W)."""
    if not IS_WINDOWS:
      self._last_action = "strafe-d-skip"
      return self._last_action
    release_key("w")
    release_key("a")
    ms = max(1.0, float(hold_ms))
    tap_key("d", hold_ms=ms)
    self._last_action = f"strafe-d-{ms:.0f}ms"
    return self._last_action

  def can_pulse(self, *, interval_ms: float | None = None) -> bool:
    gap = (
      self.final_pulse_interval_ms if interval_ms is None else float(interval_ms)
    )
    return (time.perf_counter() - self._last_pulse_at) * 1000.0 >= gap

  def tick(
    self,
    *,
    walk: bool,
    bearing_deg: float | None,
    dist_px: float,
    target_dot: float | None = None,
    fine_align: bool = False,
    close_walk: bool = False,
    force_realign: bool = False,
  ) -> str:
    if not IS_WINDOWS:
      self._last_action = "stop"
      return self._last_action

    # --- Close-walk: só W (câmera já fine-alinhada) ---
    if close_walk:
      press_key("w")
      self._last_action = "close-walk"
      return self._last_action

    # --- Fine-align: parado, só pulsos pequenos de câmera ---
    if fine_align:
      if IS_WINDOWS:
        release_key("w")
      if bearing_deg is None:
        self._last_action = "fine-wait"
        return self._last_action
      dx = self._steer_mouse(
        bearing_deg,
        max_step=self.fine_align_max_step_px,
        deadband_deg=self.fine_align_deadband_deg,
        pixels_per_deg=self.fine_align_pixels_per_deg,
        min_step=self.fine_align_min_step_px,
      )
      if dx == 0:
        self._last_action = "fine-aligned"
        return self._last_action
      if self._apply_look(dx, interval_ms=self.fine_align_look_interval_ms):
        self._last_action = f"fine-look-{dx:+d}"
      else:
        self._last_action = "fine-wait"
      return self._last_action

    if not walk or bearing_deg is None:
      return self.stop()

    abs_b = abs(bearing_deg)
    behind = target_dot is not None and target_dot < 0

    # --- Só alinhar (teste de câmera) ---
    if self.align_only:
      release_key("w")
      dx = self._steer_mouse(bearing_deg)
      if dx != 0 and self._apply_look(dx):
        self._last_action = f"look-{dx:+d}"
      elif dx != 0:
        self._last_action = "look-wait"
      else:
        self._last_action = "aligned"
      return self._last_action

    # --- Fora da faixa / alvo atrás / pós STUCK-D: para e realinha ---
    if force_realign or abs_b > self.walk_max_deg or behind:
      release_key("w")
      dx = self._steer_mouse(bearing_deg)
      if dx != 0 and self._apply_look(dx):
        self._last_action = f"realign-{dx:+d}"
      else:
        self._last_action = "realign-wait" if dx else "realign"
      return self._last_action

    # --- Perto do nó: passos de câmera mais curtos ---
    near = dist_px < 40.0
    step_cap = max(self.min_step_px, self.walk_max_step_px // 2) if near else self.walk_max_step_px
    dx = self._steer_mouse(bearing_deg, max_step=step_cap) if abs_b > self.look_deadband_deg else 0
    looked = self._apply_look(dx) if dx else False
    press_key("w")
    if looked:
      self._last_action = f"walk+look-{dx:+d}"
    elif dx:
      self._last_action = "walk+wait"
    else:
      self._last_action = "walk"
    return self._last_action

  def begin_probe_e(self) -> str:
    """Keydown E (após soltar W) — hold interruptível no brain."""
    if not IS_WINDOWS:
      self._last_action = "probe-skip"
      return self._last_action
    release_key("w")
    press_key("e")
    self._last_action = "probe-e-down"
    return self._last_action

  def end_probe_e(self) -> str:
    """Keyup E, a menos que Mining ore tenha travado o hold."""
    if not IS_WINDOWS:
      self._last_action = "probe-skip"
      return self._last_action
    if self.e_held_for_mine:
      self._last_action = "probe-e-keep"
      return self._last_action
    release_key("e")
    self._last_action = "probe-e-up"
    return self._last_action

  def keep_e_for_mine(self) -> None:
    """Mining ore: keydown E + flag — stop()/end_probe_e não soltam E."""
    if IS_WINDOWS:
      press_key("e")
    self.e_held_for_mine = True

  def clear_e_hold(self) -> None:
    """Abort/reset: solta E e limpa o flag."""
    self.e_held_for_mine = False
    if IS_WINDOWS:
      release_key("e")

  def probe_e(self, *, hold_ms: float | None = None) -> str:
    """Tap bloqueante legado; preferir begin/end_probe_e no FINAL_APPROACH."""
    if not IS_WINDOWS:
      self._last_action = "probe-skip"
      return self._last_action
    if self.e_held_for_mine:
      self._last_action = "probe-e-keep"
      return self._last_action
    release_key("w")
    ms = self.probe_e_ms if hold_ms is None else float(hold_ms)
    tap_key("e", hold_ms=max(1.0, ms))
    self._last_action = f"probe-e-{ms:.0f}ms"
    return self._last_action

  def interact(self) -> str:
    self.stop()
    if IS_WINDOWS and not self.e_held_for_mine:
      tap_key("e", hold_ms=self.interact_hold_ms)
    self._last_action = "interact-e"
    return self._last_action
