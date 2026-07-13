"""Factories de percepção — antes em mining_bot/bot.py."""

from __future__ import annotations

from typing import Any

from v2.core.config import get_hud_roi
from v2.vendor.minimap_tracker import MinimapArrowTracker
from v2.vendor.navigator import normalize_angle_deg, walk_heading_from_arrow
from v2.vendor.node_detector import MiningNodeDetector
from v2.vendor.screen_ui import MiningScreenUI


def smooth_heading_error(prev: float | None, raw: float | None, *, alpha: float) -> float | None:
  if raw is None:
    return prev
  if prev is None:
    return raw
  delta = normalize_angle_deg(raw - prev)
  if abs(delta) > 45.0:
    return raw
  return normalize_angle_deg(prev + alpha * delta)


def stable_facing_deg(
  raw: float | None,
  prev: float | None,
  *,
  max_jump_deg: float = 22.0,
) -> float | None:
  if raw is None:
    return prev
  if prev is None:
    return raw
  delta = normalize_angle_deg(raw - prev)
  if abs(delta) <= max_jump_deg:
    return raw
  flipped = normalize_angle_deg(raw + 180.0)
  flip_delta = normalize_angle_deg(flipped - prev)
  if abs(flip_delta) <= max_jump_deg:
    return flipped
  return prev


def resolve_heading_error(
  *,
  arrow,
  target,
  smooth_prev: float | None,
  smooth_alpha: float,
) -> tuple[float | None, float | None]:
  if target is None:
    return smooth_prev, smooth_prev
  raw = walk_heading_from_arrow(arrow, target.x, target.y)
  if raw is None:
    return smooth_prev, smooth_prev
  smoothed = smooth_heading_error(smooth_prev, raw, alpha=smooth_alpha)
  return smoothed, smoothed


def build_node_detector(cfg: dict[str, Any]) -> MiningNodeDetector:
  minimap = cfg["minimap"]
  center = minimap.get("player_center_ratio", {"x": 0.5, "y": 0.5})
  return MiningNodeDetector(
    tier_colors_hsv=cfg["tier_colors_hsv"],
    allowed_tiers=list(cfg.get("allowed_tiers", ["gray"])),
    road_gray_range=list(minimap["road_gray_range"]),
    player_center_ratio=(float(center["x"]), float(center["y"])),
    min_blob_area=int(cfg.get("min_blob_area", 2)),
    max_blob_area=int(cfg.get("max_blob_area", 70)),
    min_circularity=float(cfg.get("min_circularity", 0.62)),
    max_aspect_ratio=float(cfg.get("max_aspect_ratio", 1.55)),
    min_solidity=float(cfg.get("min_solidity", 0.75)),
    max_enclosing_radius_px=float(cfg.get("max_enclosing_radius_px", 9.0)),
    center_exclusion_radius_px=float(cfg.get("center_exclusion_radius_px", 18)),
    minimap_circle_margin_ratio=float(cfg.get("minimap_circle_margin_ratio", 0.06)),
    near_center_strict_radius_px=float(cfg.get("near_center_strict_radius_px", 50)),
    near_center_min_circularity=float(cfg.get("near_center_min_circularity", 0.72)),
    near_center_max_blob_area=float(cfg.get("near_center_max_blob_area", 32)),
    min_target_distance_px=float(cfg.get("min_target_distance_px", 42)),
    colored_center_exclusion_px=float(cfg.get("colored_center_exclusion_px", 8)),
    pivot_dead_zone_px=float(cfg.get("pivot_dead_zone_px", 18)),
    gray_achromatic_v_min=int(cfg.get("gray_achromatic_v_min", 185)),
    gray_achromatic_s_max=int(cfg.get("gray_achromatic_s_max", 75)),
    gray_achromatic_expand=bool(cfg.get("gray_achromatic_expand", False)),
    road_bright_protect_min=int(cfg.get("road_bright_protect_min", 200)),
  )


def build_arrow_tracker(cfg: dict[str, Any]) -> MinimapArrowTracker:
  minimap = cfg["minimap"]
  center = minimap.get("player_center_ratio", {"x": 0.5, "y": 0.5})
  return MinimapArrowTracker(
    player_center_ratio=(float(center["x"]), float(center["y"])),
    arrow_gray_min=int(minimap.get("arrow_gray_min", 145)),
    arrow_gray_max=int(minimap.get("arrow_gray_max", 175)),
    arrow_white_min=int(minimap.get("arrow_white_min", 165)),
    arrow_min_area=int(minimap.get("arrow_min_area", 8)),
    arrow_max_area=int(minimap.get("arrow_max_area", 220)),
    player_position_smoothing=float(minimap.get("player_position_smoothing", 0.62)),
    arrow_search_radius_px=float(minimap.get("arrow_search_radius_px", 42)),
    arrow_min_tip_dist_px=float(minimap.get("arrow_min_tip_dist_px", 5)),
    arrow_max_tip_dist_px=float(minimap.get("arrow_max_tip_dist_px", 30)),
    arrow_max_centroid_dist_px=float(minimap.get("arrow_max_centroid_dist_px", 22)),
    arrow_max_tip_jump_px=float(minimap.get("arrow_max_tip_jump_px", 18)),
    arrow_max_lost_frames=int(minimap.get("arrow_max_lost_frames", 36)),
    fixed_player_anchor=bool(minimap.get("fixed_player_anchor", True)),
    player_center_calibrated=bool(minimap.get("player_center_calibrated", False)),
  )


def build_screen_ui(cfg: dict[str, Any]) -> MiningScreenUI:
  return MiningScreenUI(
    progress_roi=get_hud_roi(cfg),
    white_threshold=int(cfg.get("progress_white_threshold", 175)),
    min_bar_pixels=int(cfg.get("min_bar_pixels", 120)),
    requires_label=bool(cfg.get("mining_requires_label", True)),
  )
