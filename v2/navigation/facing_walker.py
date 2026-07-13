"""
Navegação v2 — Facing Pursuit.

Estratégia simples (sem pulsos PursuitWalker):
  |erro| grande  → segura A ou D (só gira)
  |erro| médio   → W + A/D (curva suave)
  |erro| pequeno → W reto

NavProgressTracker corrige inversão A/D automaticamente.
"""

from __future__ import annotations

from typing import Any

from v2.vendor.keyboard_input import IS_WINDOWS, press_key, release_all_keys, release_key, tap_key
from progress_nav import NavProgressTracker


class FacingWalkController:
  def __init__(self, cfg: dict[str, Any]) -> None:
    nav = cfg.get("navigation", {})
    interact = cfg.get("interact", {})
    self.align_deg = float(nav.get("align_deg", 12))
    self.walk_deg = float(nav.get("turn_walk_deg", 30))
    self.turn_only_deg = float(nav.get("turn_only_deg", 55))
    self.interact_hold_ms = float(
      cfg.get("walker", {}).get("interact_hold_ms", 80)
    )
    self.e_held_for_mine = False
    self.progress = NavProgressTracker(
      min_walk_gain_px=float(nav.get("min_walk_gain_px", 0.5)),
      min_turn_gain_deg=float(nav.get("min_turn_gain_deg", 2.0)),
      invert_streak=int(nav.get("invert_streak", 3)),
    )
    self._last_action = "idle"
    self._last_steer: str | None = None

  @property
  def last_action(self) -> str:
    return self._last_action

  def reset_progress(self) -> None:
    self.progress.reset()
    self._last_steer = None

  def stop(self) -> str:
    if IS_WINDOWS:
      if self.e_held_for_mine:
        for key in ("w", "a", "d", "s"):
          release_key(key)
      else:
        release_all_keys()
    self._last_steer = None
    self._last_action = "idle"
    return self._last_action

  def observe(self, *, dist_px: float, bearing_deg: float) -> None:
    self.progress.observe(
      tile_dist_px=dist_px,
      screen_dist_px=dist_px,
      heading_error_deg=bearing_deg,
    )

  def feedback(self, bearing_before: float | None, bearing_after: float | None) -> None:
    if (
      bearing_before is None
      or bearing_after is None
      or self._last_steer not in ("a", "d")
    ):
      return
    self.progress.feedback_turn(bearing_before, bearing_after, self._last_steer)

  def _release_steer(self) -> None:
    if IS_WINDOWS:
      release_key("a")
      release_key("d")

  def _hold_steer(self, key: str) -> None:
    if not IS_WINDOWS:
      return
    other = "d" if key == "a" else "a"
    release_key(other)
    press_key(key)
    self._last_steer = key

  def tick(
    self,
    *,
    walk: bool,
    bearing_deg: float | None,
    dist_px: float,
    target_dot: float | None = None,
  ) -> str:
    if not IS_WINDOWS:
      self._last_action = "stop"
      return self._last_action

    if not walk or bearing_deg is None:
      return self.stop()

    err = self.progress.control_heading(bearing_deg, target_dot=target_dot)
    abs_err = abs(err)

    if IS_WINDOWS:
      release_key("w")
    self._release_steer()

    # Alvo atrás da seta — só gira, nunca W.
    if target_dot is not None and target_dot < 0:
      key = "d" if err > 0 else "a"
      self._hold_steer(key)
      self._last_action = f"behind-turn-{key}"
      return self._last_action

    if abs_err >= self.turn_only_deg:
      key = "d" if err > 0 else "a"
      self._hold_steer(key)
      self._last_action = f"turn-{key}"
      return self._last_action

    if abs_err <= self.walk_deg and self.progress.can_walk(err):
      if IS_WINDOWS:
        press_key("w")
      if abs_err > self.align_deg:
        key = "d" if err > 0 else "a"
        self._hold_steer(key)
        self._last_action = f"walk-{key}"
      else:
        self._last_steer = None
        self._last_action = "walk"
      if self._last_action.startswith("walk"):
        self.progress.feedback_walk()
      return self._last_action

    key = "d" if err > 0 else "a"
    self._hold_steer(key)
    self._last_action = f"align-{key}"
    return self._last_action

  def begin_probe_e(self) -> str:
    if IS_WINDOWS:
      release_key("w")
      press_key("e")
    self._last_action = "probe-e-down"
    return self._last_action

  def end_probe_e(self) -> str:
    if IS_WINDOWS and not self.e_held_for_mine:
      release_key("e")
      self._last_action = "probe-e-up"
    else:
      self._last_action = "probe-e-keep"
    return self._last_action

  def keep_e_for_mine(self) -> None:
    """Trava hold e garante keydown E (idempotente se já down)."""
    if IS_WINDOWS:
      press_key("e")
    self.e_held_for_mine = True

  def clear_e_hold(self) -> None:
    self.e_held_for_mine = False
    if IS_WINDOWS:
      release_key("e")

  def probe_e(self, *, hold_ms: float | None = None) -> str:
    if IS_WINDOWS and not self.e_held_for_mine:
      ms = self.interact_hold_ms if hold_ms is None else float(hold_ms)
      tap_key("e", hold_ms=max(1.0, ms))
    self._last_action = "probe-e"
    return self._last_action

  def interact(self) -> str:
    self.stop()
    if IS_WINDOWS and not self.e_held_for_mine:
      tap_key("e", hold_ms=self.interact_hold_ms)
    self._last_action = "interact-e"
    return self._last_action

  def pulse_space(self, *, hold_ms: float = 150.0) -> str:
    """STUCK recovery step A: tap Space (jump) before D strafe."""
    if not IS_WINDOWS:
      self._last_action = "space-skip"
      return self._last_action
    release_key("w")
    self._release_steer()
    ms = max(1.0, float(hold_ms))
    tap_key("space", hold_ms=ms)
    self._last_steer = None
    self._last_action = f"space-{ms:.0f}ms"
    return self._last_action

  def pulse_strafe_d(self, *, hold_ms: float = 2000.0) -> str:
    """STUCK recovery: solta W/A, segura D, solta D."""
    if not IS_WINDOWS:
      self._last_action = "strafe-d-skip"
      return self._last_action
    release_key("w")
    self._release_steer()
    ms = max(1.0, float(hold_ms))
    tap_key("d", hold_ms=ms)
    self._last_steer = None
    self._last_action = f"strafe-d-{ms:.0f}ms"
    return self._last_action
