"""Blips cinza no minimapa."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from v2.core.types import ArrowState, Blip, TargetLock


def _node_to_blip(node: Any) -> Blip:
  return Blip(
    x=float(node.x),
    y=float(node.y),
    tier=node.tier.lower(),
    radius=float(node.radius),
    distance_px=float(node.distance_px),
  )


def detect_blips(
  minimap_bgr: np.ndarray,
  *,
  detector: Any,
  arrow_tracker: Any,
  arrow: ArrowState,
  legacy_arrow: Any,
  allowed_tiers: list[str],
  min_distance_px: float,
  lock: TargetLock | None = None,
) -> tuple[Blip, ...]:
  has_lock = lock is not None
  # Com lock, reduz exclusão central — blip colado no pivot ainda aparece.
  excl_radius = 8.0 if has_lock else 22.0
  excl = arrow_tracker.build_exclusion_mask(
    minimap_bgr.shape,
    legacy_arrow,
    center_radius_px=excl_radius,
  )
  track_lock = SimpleNamespace(tier=lock.tier) if lock is not None else None
  scan = detector.scan_blips(
    minimap_bgr,
    player_exclusion_mask=excl,
    player_x=arrow.pivot_x,
    player_y=arrow.pivot_y,
    min_distance_px=min_distance_px,
    track_lock=track_lock,
  )
  tiers = {t.lower() for t in allowed_tiers}
  merged: list[Any] = list(scan.nodes)
  if scan.track_nodes:
    for tracked in scan.track_nodes:
      if any(math.hypot(tracked.x - node.x, tracked.y - node.y) < 6.0 for node in merged):
        continue
      merged.append(tracked)
  out: list[Blip] = []
  for node in merged:
    if node.tier.lower() not in tiers:
      continue
    out.append(_node_to_blip(node))
  return tuple(out)
