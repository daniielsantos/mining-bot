"""Contratos de dados — v2. Imutáveis por frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Phase(str, Enum):
  SCAN = "SCAN"
  GOTO = "GOTO"
  FINAL_APPROACH = "FINAL_APPROACH"
  READY_INTERACT = "READY_INTERACT"
  INTERACT = "INTERACT"
  MINING = "MINING"
  COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class Blip:
  """Nó detectado no minimapa — coordenadas de TELA (px)."""

  x: float
  y: float
  tier: str
  radius: float
  distance_px: float  # distância ao pivot


@dataclass(frozen=True)
class ArrowState:
  pivot_x: float
  pivot_y: float
  tip_x: float | None
  tip_y: float | None
  facing_deg: float | None
  detected: bool


@dataclass(frozen=True)
class HudState:
  mining_active: bool
  progress_pct: float | None
  has_label: bool


@dataclass
class TargetLock:
  """
  Alvo travado — só existe na tela.
  track_id: identificador estável entre frames (matching por proximidade).
  pinned: blip sumiu perto do pivot — mantém posição até chegar/interagir.
  """

  track_id: int
  x: float
  y: float
  tier: str
  lost_frames: int = 0
  done: bool = False
  pinned: bool = False


@dataclass
class FrameContext:
  """
  Snapshot de um tick. Perception preenche visão; brain preenche decisão.
  """

  tick: int
  timestamp: float
  minimap_bgr: np.ndarray
  hud_bgr: np.ndarray
  pivot: tuple[float, float]
  arrow: ArrowState
  blips: tuple[Blip, ...]
  hud: HudState
  interaction_bgr: np.ndarray | None = None
  lock: TargetLock | None = None
  phase: Phase = Phase.SCAN
  bearing_deg: float | None = None
  dist_px: float = 0.0
  aligned: bool = False
  arrived: bool = False
  action: str = "idle"
  meta: dict[str, Any] = field(default_factory=dict)

  def with_updates(self, **kwargs) -> FrameContext:
    from dataclasses import replace

    return replace(self, **kwargs)

  def debug_dict(self) -> dict[str, Any]:
    lock = self.lock
    payload: dict[str, Any] = {
      "tick": self.tick,
      "phase": self.phase.value,
      "action": self.action,
      "bearing_deg": self.bearing_deg,
      "dist_px": self.dist_px,
      "aligned": self.aligned,
      "arrived": self.arrived,
      "lock_id": lock.track_id if lock else None,
      "lock_pinned": lock.pinned if lock else False,
      "lock_x": lock.x if lock else None,
      "lock_y": lock.y if lock else None,
      "blips": len(self.blips),
      "arrow_ok": self.arrow.detected,
      "facing_deg": self.arrow.facing_deg,
      "pivot_x": self.pivot[0],
      "pivot_y": self.pivot[1],
      "mining_active": self.hud.mining_active,
      "progress_pct": self.hud.progress_pct,
    }
    for key, value in self.meta.items():
      if key in ("legacy_arrow", "hud_result"):
        continue
      if isinstance(value, (str, int, float, bool)) or value is None:
        payload[key] = value
    return payload
