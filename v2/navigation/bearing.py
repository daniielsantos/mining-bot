"""Rumo alvo — facing_deg estável + debug visual da ponta."""

from __future__ import annotations

import math
from typing import Any

from v2.vendor.navigator import facing_relative_error

from v2.core.types import ArrowState

# Em 1a pessoa a seta do minimapa NUNCA gira: sempre aponta para cima na tela.
# (y cresce para baixo → frente = -90° em atan2)
FIRST_PERSON_FORWARD_DEG = -90.0


def forward_heading_deg(
  arrow: ArrowState,
  legacy_arrow: Any | None = None,
  *,
  first_person: bool = False,
) -> float | None:
  """
  Frente do personagem.
  1a pessoa: sempre cima na tela (seta fixa).
  3a pessoa: pivot → ponta (nariz detectado).
  """
  if first_person:
    return FIRST_PERSON_FORWARD_DEG

  px = arrow.pivot_x
  py = arrow.pivot_y
  tip_x = arrow.tip_x
  tip_y = arrow.tip_y
  if (tip_x is None or tip_y is None) and legacy_arrow is not None:
    tip_x = getattr(legacy_arrow, "arrow_tip_x", None)
    tip_y = getattr(legacy_arrow, "arrow_tip_y", None)
    px, py = legacy_arrow.pivot()
  if tip_x is not None and tip_y is not None:
    if math.hypot(tip_x - px, tip_y - py) >= 2.0:
      return math.degrees(math.atan2(tip_y - py, tip_x - px))
  if arrow.facing_deg is not None:
    return float(arrow.facing_deg)
  if legacy_arrow is not None:
    facing = getattr(legacy_arrow, "arrow_angle_deg", None)
    if facing is not None:
      return float(facing)
  return None


def camera_align_bearing(
  arrow: ArrowState,
  legacy_arrow: Any,
  target_x: float,
  target_y: float,
  *,
  prefer_facing: bool = True,
  first_person: bool = False,
) -> float | None:
  """
  Quanto girar a camera: angulo do alvo relativo a frente.
  1a pessoa: frente = cima; o mapa/blips giram em volta da seta.
  Sem fold ±90° — caminho curto em ±180°.
  """
  del prefer_facing
  fwd = forward_heading_deg(arrow, legacy_arrow, first_person=first_person)
  if fwd is None:
    return None
  return facing_relative_error(
    arrow.pivot_x,
    arrow.pivot_y,
    target_x,
    target_y,
    fwd,
  )


def target_ahead_dot_facing(
  arrow: ArrowState,
  target_x: float,
  target_y: float,
  *,
  legacy_arrow: Any | None = None,
  first_person: bool = False,
) -> float | None:
  """Alvo a frente da linha amarela? >0 = sim."""
  fwd = forward_heading_deg(arrow, legacy_arrow, first_person=first_person)
  if fwd is None:
    return None
  rad = math.radians(fwd)
  fx, fy = math.cos(rad), math.sin(rad)
  tx = target_x - arrow.pivot_x
  ty = target_y - arrow.pivot_y
  return fx * tx + fy * ty


def bearing_to_target(
  arrow: ArrowState,
  target_x: float,
  target_y: float,
  *,
  legacy_arrow: Any | None = None,
  first_person: bool = False,
) -> float | None:
  """Quanto girar para o blip ficar à frente da seta (linha amarela)."""
  fwd = forward_heading_deg(arrow, legacy_arrow, first_person=first_person)
  if fwd is None:
    return None
  return facing_relative_error(
    arrow.pivot_x,
    arrow.pivot_y,
    target_x,
    target_y,
    fwd,
  )


def target_ahead_dot(
  arrow: ArrowState,
  target_x: float,
  target_y: float,
  *,
  legacy_arrow: Any | None = None,
) -> float | None:
  """Dot pivot→ponta · pivot→alvo. >0 = alvo à frente."""
  if legacy_arrow is None:
    return None
  tip_x = getattr(legacy_arrow, "arrow_tip_x", None)
  tip_y = getattr(legacy_arrow, "arrow_tip_y", None)
  if tip_x is None or tip_y is None:
    return None
  px, py = arrow.pivot_x, arrow.pivot_y
  fx = float(tip_x) - px
  fy = float(tip_y) - py
  fl = (fx * fx + fy * fy) ** 0.5
  if fl < 2.0:
    return None
  tx = target_x - px
  ty = target_y - py
  return (fx * tx + fy * ty) / fl


def first_person_tip(
  pivot_x: float,
  pivot_y: float,
  *,
  length: float = 12.0,
) -> tuple[float, float]:
  """Ponta visual da seta fixa (sempre para cima na tela)."""
  rad = math.radians(FIRST_PERSON_FORWARD_DEG)
  return pivot_x + math.cos(rad) * length, pivot_y + math.sin(rad) * length
