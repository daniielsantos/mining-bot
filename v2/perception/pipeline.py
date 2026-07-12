"""Monta FrameContext de percepção."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from node_detector import TargetLock as V1TargetLock
from v2.core.types import Blip, FrameContext, HudState, Phase
from v2.perception.arrow import detect_arrow
from v2.perception.hud import detect_hud


def _scan_to_blips(scan: Any, *, allowed: set[str]) -> tuple[Blip, ...]:
  """nodes + track_nodes (unlock perto do pivô) → overlay/lock veem o mesmo centro."""
  merged: list[Any] = list(scan.nodes)
  track = getattr(scan, "track_nodes", None) or []
  for tracked in track:
    if any(math.hypot(tracked.x - node.x, tracked.y - node.y) < 6.0 for node in merged):
      continue
    merged.append(tracked)
  out: list[Blip] = []
  for node in merged:
    if node.tier.lower() not in allowed:
      continue
    out.append(
      Blip(
        x=float(node.x),
        y=float(node.y),
        tier=node.tier.lower(),
        radius=float(node.radius),
        distance_px=float(node.distance_px),
      )
    )
  return tuple(out)


def perceive(
  tick: int,
  minimap_bgr: np.ndarray,
  hud_bgr: np.ndarray,
  *,
  arrow_tracker: Any,
  node_detector: Any,
  screen_ui: Any,
  cfg: dict,
  phase: Phase = Phase.SCAN,
  v1_lock: V1TargetLock | None = None,
  interaction_bgr: np.ndarray | None = None,
  mining_ore: Any | None = None,
) -> FrameContext:
  nav = cfg.get("navigation", {})
  first_person = bool(nav.get("camera", {}).get("first_person", True))
  if str(nav.get("control_mode", "camera")).lower() == "camera":
    first_person = True
  arrow, legacy_arrow = detect_arrow(
    minimap_bgr, arrow_tracker, first_person=first_person
  )
  min_pick = float(nav.get("min_pick_px", 18))
  has_lock = v1_lock is not None
  # Com lock: exclui a seta (~10px) pra nao detectar a seta como no cinza.
  # SCAN: exclusão curta (~14). 40px escondia o disco perto do pivô e o bot
  # só via fagulhas longe → lock ghost → abandon → SCAN thrash.
  excl_default = float(nav.get("arrow_exclusion_radius_px", 14))
  excl_radius = 10.0 if has_lock else float(
    cfg.get("arrow_exclusion_radius_px", excl_default)
  )
  excl = arrow_tracker.build_exclusion_mask(
    minimap_bgr.shape,
    legacy_arrow,
    center_radius_px=excl_radius,
  )
  scan = node_detector.scan_blips(
    minimap_bgr,
    player_exclusion_mask=excl,
    player_x=arrow.pivot_x,
    player_y=arrow.pivot_y,
    min_distance_px=0.0 if has_lock else min_pick,
    track_lock=v1_lock,
  )
  allowed = {t.lower() for t in cfg.get("allowed_tiers", ["gray"])}
  blips = _scan_to_blips(scan, allowed=allowed)
  hud, hud_result = detect_hud(hud_bgr, screen_ui)
  ore_score = 0.0
  ore_found = False
  ore_thresh = 0.85
  ore_near = False
  if mining_ore is not None:
    ore_hit = mining_ore.detect(hud_bgr)
    ore_score = float(ore_hit.score)
    ore_found = bool(ore_hit.found)
    ore_near = bool(getattr(ore_hit, "near_miss", False))
    ore_thresh = float(
      getattr(
        mining_ore,
        "hold_min",
        getattr(mining_ore, "threshold", 0.85),
      )
    )
    if ore_found or ore_score >= ore_thresh:
      from screen_ui import MiningUIResult

      hud = HudState(
        mining_active=ore_found or hud.mining_active,
        progress_pct=hud.progress_pct,
        has_label=True,
      )
      hud_result = MiningUIResult(
        mining_active=ore_found or bool(hud_result.mining_active),
        progress_pct=hud_result.progress_pct,
        interaction_hint=bool(hud_result.interaction_hint),
        has_label=True,
      )
  pivot = (arrow.pivot_x, arrow.pivot_y)
  return FrameContext(
    tick=tick,
    timestamp=time.perf_counter(),
    minimap_bgr=minimap_bgr,
    hud_bgr=hud_bgr,
    interaction_bgr=interaction_bgr,
    pivot=pivot,
    arrow=arrow,
    blips=blips,
    hud=hud,
    phase=phase,
    meta={
      "legacy_arrow": legacy_arrow,
      "hud_result": hud_result,
      "scan": scan,
      "first_person": first_person,
      "mining_ore_score": ore_score,
      "mining_ore_found": ore_found,
      "mining_ore_threshold": ore_thresh,
      "mining_ore_near_miss": ore_near,
    },
  )
