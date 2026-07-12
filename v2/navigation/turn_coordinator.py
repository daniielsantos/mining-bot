"""
Coordenação de giro no minimapa rotativo.

Enquanto |bearing| é grande, o mapa gira e dist_px na tela MENTE
(o blip se move mesmo sem andar). Aqui:
  - compromete lado A/D por um tempo mínimo
  - só libera W quando bearing pequeno E estável por N frames
  - mede progresso pelo |bearing| cair, não pela distância na tela
"""

from __future__ import annotations

import time


class TurnCoordinator:
  def __init__(
    self,
    *,
    align_deg: float = 12.0,
    settle_frames: int = 4,
    commit_ms: float = 320.0,
    flip_deg: float = 28.0,
  ) -> None:
    self.align_deg = align_deg
    self.settle_frames = settle_frames
    self.commit_ms = commit_ms
    self.flip_deg = flip_deg
    self._committed: str | None = None
    self._commit_until = 0.0
    self._align_streak = 0
    self._phase = "idle"
    self._last_abs_bearing: float | None = None

  def reset(self) -> None:
    self._committed = None
    self._commit_until = 0.0
    self._align_streak = 0
    self._phase = "idle"
    self._last_abs_bearing = None

  @property
  def phase(self) -> str:
    return self._phase

  def update(self, bearing_deg: float | None) -> str:
    """
    Retorna fase: idle | align-a | align-d | settle | walk
    """
    if bearing_deg is None:
      self._phase = "idle"
      self._align_streak = 0
      return self._phase

    abs_b = abs(bearing_deg)
    now = time.perf_counter()
    desired = "d" if bearing_deg > 0 else "a"

    if abs_b <= self.align_deg:
      self._align_streak += 1
      self._committed = None
      if self._align_streak >= self.settle_frames:
        self._phase = "walk"
      else:
        self._phase = "settle"
      self._last_abs_bearing = abs_b
      return self._phase

    self._align_streak = 0
    if self._committed is None or now >= self._commit_until:
      self._committed = desired
      self._commit_until = now + self.commit_ms / 1000.0
    elif desired != self._committed:
      # Só troca de lado se o erro inverter forte (evita flip durante rotação rápida).
      if abs_b >= self.flip_deg:
        self._committed = desired
        self._commit_until = now + self.commit_ms / 1000.0

    self._phase = f"align-{self._committed}"
    self._last_abs_bearing = abs_b
    return self._phase

  def steering_bearing(self, bearing_deg: float | None) -> float | None:
    """Força sinal coerente com o lado comprometido durante align."""
    if bearing_deg is None or not self._phase.startswith("align-"):
      return bearing_deg
    side = self._phase.split("-", 1)[1]
    mag = max(abs(bearing_deg), 8.0)
    return mag if side == "d" else -mag

  def bearing_progress(self, bearing_deg: float | None) -> float | None:
    """Quanto o |bearing| caiu desde o frame anterior (positivo = melhorou)."""
    if bearing_deg is None or self._last_abs_bearing is None:
      return None
    prev = self._last_abs_bearing
    cur = abs(bearing_deg)
    return prev - cur

  def is_turning(self) -> bool:
    return self._phase.startswith("align-")
