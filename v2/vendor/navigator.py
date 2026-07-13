"""Helpers de rumo usados pelo pursuit (subset do navigator v1)."""

from __future__ import annotations

import math


def normalize_angle_deg(angle: float) -> float:
  while angle > 180.0:
    angle -= 360.0
  while angle < -180.0:
    angle += 360.0
  return angle


def arrow_to_target_error(
  anchor_x: float,
  anchor_y: float,
  tip_x: float,
  tip_y: float,
  target_x: float,
  target_y: float,
) -> float:
  fwd_x = tip_x - anchor_x
  fwd_y = tip_y - anchor_y
  length = math.hypot(fwd_x, fwd_y)
  if length < 1.0:
    return 0.0
  ux, uy = fwd_x / length, fwd_y / length
  to_x = target_x - anchor_x
  to_y = target_y - anchor_y
  cross = ux * to_y - uy * to_x
  dot = ux * to_x + uy * to_y
  if abs(cross) < 1e-6 and abs(dot) < 1e-6:
    return 0.0
  return normalize_angle_deg(math.degrees(math.atan2(cross, dot)))


def facing_relative_error(
  anchor_x: float,
  anchor_y: float,
  target_x: float,
  target_y: float,
  facing_deg: float,
) -> float:
  to_target = math.degrees(
    math.atan2(target_y - anchor_y, target_x - anchor_x)
  )
  return normalize_angle_deg(to_target - facing_deg)


def walk_heading_from_arrow(
  arrow,
  target_x: float,
  target_y: float,
) -> float | None:
  px, py = arrow.pivot()
  tip_x = arrow.arrow_tip_x
  tip_y = arrow.arrow_tip_y
  if tip_x is None or tip_y is None:
    facing = getattr(arrow, "arrow_angle_deg", None)
    if facing is not None:
      return facing_relative_error(px, py, target_x, target_y, facing)
    return None
  if math.hypot(tip_x - px, tip_y - py) < 3.0:
    facing = getattr(arrow, "arrow_angle_deg", None)
    if facing is not None:
      return facing_relative_error(px, py, target_x, target_y, facing)
    return None

  def heading_with_tip(tx: float, ty: float) -> float:
    return arrow_to_target_error(px, py, tx, ty, target_x, target_y)

  err_a = heading_with_tip(tip_x, tip_y)
  flip_x = px - (tip_x - px)
  flip_y = py - (tip_y - py)
  err_b = heading_with_tip(flip_x, flip_y)

  facing = getattr(arrow, "arrow_angle_deg", None)
  if facing is None:
    facing = getattr(arrow, "facing_deg", None)

  if facing is not None:
    face_err = facing_relative_error(px, py, target_x, target_y, facing)
    flip_delta = abs(normalize_angle_deg(err_a - err_b))
    if flip_delta > 155.0:
      return face_err
    pick_a = abs(normalize_angle_deg(err_a - face_err))
    pick_b = abs(normalize_angle_deg(err_b - face_err))
    best = err_a if pick_a <= pick_b else err_b
    if abs(normalize_angle_deg(best - face_err)) > 35.0:
      return face_err
    return best

  if abs(err_b) + 2.0 < abs(err_a):
    return err_b
  if abs(err_a) > 95.0 and abs(err_b) < abs(err_a):
    return err_b
  return err_a


def camera_heading_from_arrow(
  arrow,
  target_x: float,
  target_y: float,
) -> float | None:
  px, py = arrow.pivot()
  tip_x = arrow.arrow_tip_x
  tip_y = arrow.arrow_tip_y
  tgt_a = math.degrees(math.atan2(target_y - py, target_x - px))

  if tip_x is None or tip_y is None or math.hypot(tip_x - px, tip_y - py) < 3.0:
    facing = getattr(arrow, "arrow_angle_deg", None)
    if facing is None:
      facing = getattr(arrow, "facing_deg", None)
    if facing is None:
      facing = -90.0
    return normalize_angle_deg(tgt_a - facing)

  tip_a = math.degrees(math.atan2(tip_y - py, tip_x - px))
  return normalize_angle_deg(tgt_a - tip_a)
