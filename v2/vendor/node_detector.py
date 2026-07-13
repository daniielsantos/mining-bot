from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class MiningNode:
  tier: str
  x: float
  y: float
  radius: float
  area: float
  distance_px: float
  circularity: float
  ghost: bool = False
  virtual: bool = False
  node_id: int | None = None


@dataclass
class _NodeTrack:
  node_id: int
  tier: str
  x: float
  y: float
  vel_x: float
  vel_y: float
  area: float
  anchor_offsets: tuple[tuple[float, float, str], ...]
  lost_frames: int = 0


class NodeRegistry:
  """IDs estaveis por no — identidade via ancora + predicao entre frames."""

  def __init__(self) -> None:
    self._next_id = 1
    self._tracks: dict[int, _NodeTrack] = {}
    self._pinned: set[int] = set()

  def get(self, node_id: int) -> _NodeTrack | None:
    return self._tracks.get(node_id)

  def pin(self, node_id: int) -> None:
    if node_id > 0:
      self._pinned.add(node_id)

  def unpin(self, node_id: int) -> None:
    self._pinned.discard(node_id)

  def forget(self, node_id: int) -> None:
    self._pinned.discard(node_id)
    self._tracks.pop(node_id, None)

  def assign(
    self,
    detections: list[MiningNode],
    anchor_pool: list[MiningNode],
    *,
    build_anchors,
    anchor_mismatch,
  ) -> list[MiningNode]:
    if not detections:
      return []

    pairs: list[tuple[float, int, int]] = []
    for track_id, track in self._tracks.items():
      pred_x = track.x + track.vel_x
      pred_y = track.y + track.vel_y
      pinned = track_id in self._pinned
      max_dist = (12.0 + track.lost_frames * 2.5) if pinned else (18.0 + track.lost_frames * 5.0)
      max_score = 20.0 if pinned else 38.0
      for index, det in enumerate(detections):
        if det.tier != track.tier:
          continue
        dist = math.hypot(det.x - pred_x, det.y - pred_y)
        if dist > max_dist:
          continue
        if track.anchor_offsets:
          mismatch = anchor_mismatch(det, anchor_pool, track.anchor_offsets)
          if pinned and mismatch > 18.0:
            continue
          score = mismatch + dist * 0.35
        else:
          score = dist
        pairs.append((score, track_id, index))

    pairs.sort(key=lambda item: item[0])
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    assignments: dict[int, int] = {}
    for score, track_id, index in pairs:
      if track_id in used_tracks or index in used_dets:
        continue
      max_score = 20.0 if track_id in self._pinned else 38.0
      if score > max_score:
        continue
      assignments[index] = track_id
      used_tracks.add(track_id)
      used_dets.add(index)

    tagged: list[MiningNode] = []
    for index, det in enumerate(detections):
      if index in assignments:
        track_id = assignments[index]
        track = self._tracks[track_id]
        obs_vx = det.x - track.x
        obs_vy = det.y - track.y
        track.vel_x = 0.6 * obs_vx + 0.4 * track.vel_x
        track.vel_y = 0.6 * obs_vy + 0.4 * track.vel_y
        track.x = det.x
        track.y = det.y
        track.area = det.area
        track.lost_frames = 0
        tagged.append(replace(det, node_id=track_id))
      else:
        track_id = self._next_id
        self._next_id += 1
        anchors = build_anchors(det, anchor_pool)
        self._tracks[track_id] = _NodeTrack(
          node_id=track_id,
          tier=det.tier,
          x=det.x,
          y=det.y,
          vel_x=0.0,
          vel_y=0.0,
          area=det.area,
          anchor_offsets=anchors,
        )
        tagged.append(replace(det, node_id=track_id))

    for track_id, track in list(self._tracks.items()):
      if track_id not in used_tracks:
        track.lost_frames += 1
        if track.lost_frames > 150:
          del self._tracks[track_id]

    return tagged

  def tag_untracked(
    self,
    detections: list[MiningNode],
    known: list[MiningNode],
    anchor_pool: list[MiningNode],
    *,
    build_anchors,
    anchor_mismatch,
    merge_radius_px: float = 8.0,
  ) -> list[MiningNode]:
    tagged: list[MiningNode] = []
    for det in detections:
      match = None
      best_dist = merge_radius_px
      for known_node in known:
        if known_node.node_id is None or known_node.tier != det.tier:
          continue
        dist = math.hypot(det.x - known_node.x, det.y - known_node.y)
        if dist < best_dist:
          best_dist = dist
          match = known_node
      if match is not None:
        tagged.append(replace(det, node_id=match.node_id))
        continue
      assigned = self.assign([det], anchor_pool, build_anchors=build_anchors, anchor_mismatch=anchor_mismatch)
      tagged.append(assigned[0])
      known.append(assigned[0])
    return tagged


@dataclass
class NodeScanResult:
  nodes: list[MiningNode]
  target: MiningNode | None
  masks: dict[str, np.ndarray]
  player_x: float
  player_y: float
  # Deteccao do tier travado sem mascara da seta (blip perto do jogador).
  track_nodes: list[MiningNode] | None = None
  track_masks: dict[str, np.ndarray] | None = None


@dataclass
class TargetLock:
  """Um unico alvo travado por node_id ate o usuario pressionar E/F8."""

  tier: str
  locked_x: float
  locked_y: float
  pick_distance_px: float
  last_distance_px: float
  node_id: int = 0
  locked_area: float = 0.0
  last_bearing_deg: float = 0.0
  vel_x: float = 0.0
  vel_y: float = 0.0
  lost_frames: int = 0
  committed: bool = False
  min_seen_distance_px: float = 999.0
  approached_outside: bool = False
  pick_x: float = 0.0
  pick_y: float = 0.0
  virtual_x: float = 0.0
  virtual_y: float = 0.0
  # Ultima posicao virtual fora da zona da seta — evita colapsar no pivot.
  last_outside_x: float = 0.0
  last_outside_y: float = 0.0
  # Offsets para outros blips no instante da trava — sobrevivem ao mapa rolar.
  anchor_offsets: tuple[tuple[float, float, str], ...] = ()


def normalize_angle_deg(angle: float) -> float:
  while angle > 180.0:
    angle -= 360.0
  while angle < -180.0:
    angle += 360.0
  return angle


def bearing_deg(origin_x: float, origin_y: float, point_x: float, point_y: float) -> float:
  return math.degrees(math.atan2(point_y - origin_y, point_x - origin_x))


class MiningNodeDetector:
  """Detecta blips de mineracao (cinza/laranja/turquesa) no minimapa."""

  def __init__(
    self,
    *,
    tier_colors_hsv: dict[str, dict[str, list[int]]],
    allowed_tiers: list[str],
    road_gray_range: list[int],
    player_center_ratio: tuple[float, float] = (0.5, 0.5),
    min_blob_area: int = 2,
    max_blob_area: int = 70,
    min_circularity: float = 0.62,
    max_aspect_ratio: float = 1.55,
    min_solidity: float = 0.75,
    max_enclosing_radius_px: float = 9.0,
    center_exclusion_radius_px: float = 18.0,
    minimap_circle_margin_ratio: float = 0.06,
    min_distance_from_center_px: float = 6.0,
    near_center_strict_radius_px: float = 50.0,
    near_center_min_circularity: float = 0.72,
    near_center_max_blob_area: float = 32.0,
    min_target_distance_px: float = 42.0,
    colored_center_exclusion_px: float = 8.0,
    pivot_dead_zone_px: float = 18.0,
    gray_achromatic_v_min: int = 185,
    gray_achromatic_s_max: int = 75,
    gray_achromatic_expand: bool = False,
    road_bright_protect_min: int = 200,
  ) -> None:
    self.tier_colors_hsv = tier_colors_hsv
    self.allowed_tiers = [t.lower() for t in allowed_tiers]
    self.road_gray_min = int(road_gray_range[0])
    self.road_gray_max = int(road_gray_range[1])
    self.player_center_ratio = player_center_ratio
    self.min_blob_area = min_blob_area
    self.max_blob_area = max_blob_area
    self.min_circularity = min_circularity
    self.max_aspect_ratio = float(max_aspect_ratio)
    self.min_solidity = float(min_solidity)
    self.max_enclosing_radius_px = float(max_enclosing_radius_px)
    self.center_exclusion_radius_px = center_exclusion_radius_px
    self.minimap_circle_margin_ratio = minimap_circle_margin_ratio
    self.min_distance_from_center_px = min_distance_from_center_px
    self.near_center_strict_radius_px = near_center_strict_radius_px
    self.near_center_min_circularity = near_center_min_circularity
    self.near_center_max_blob_area = near_center_max_blob_area
    self.min_target_distance_px = min_target_distance_px
    self.colored_center_exclusion_px = colored_center_exclusion_px
    self.pivot_dead_zone_px = pivot_dead_zone_px
    self.gray_achromatic_v_min = int(gray_achromatic_v_min)
    self.gray_achromatic_s_max = int(gray_achromatic_s_max)
    # Default off: tier_colors_hsv.gray is the source of truth (calibrate once).
    self.gray_achromatic_expand = bool(gray_achromatic_expand)
    self.road_bright_protect_min = int(road_bright_protect_min)
    self.node_registry = NodeRegistry()

  def _anchor_pool(self, result: NodeScanResult) -> list[MiningNode]:
    pool = list(result.nodes)
    if result.track_nodes:
      for tracked in result.track_nodes:
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0 for node in pool
        ):
          pool.append(tracked)
    return pool

  def _tag_scan_nodes(self, result: NodeScanResult) -> NodeScanResult:
    pool = self._anchor_pool(result)
    tagged_nodes = self.node_registry.assign(
      result.nodes,
      pool,
      build_anchors=self._build_anchors,
      anchor_mismatch=self._anchor_mismatch,
    )
    tagged_track = None
    if result.track_nodes:
      tagged_track = self.node_registry.tag_untracked(
        result.track_nodes,
        list(tagged_nodes),
        pool,
        build_anchors=self._build_anchors,
        anchor_mismatch=self._anchor_mismatch,
      )
    target = result.target
    if target is not None:
      for node in tagged_nodes:
        if (
          node.tier == target.tier
          and math.hypot(node.x - target.x, node.y - target.y) < 6.0
        ):
          target = node
          break
    return NodeScanResult(
      nodes=tagged_nodes,
      target=target,
      masks=result.masks,
      player_x=result.player_x,
      player_y=result.player_y,
      track_nodes=tagged_track,
      track_masks=result.track_masks,
    )

  def _find_node_by_id(
    self,
    result: NodeScanResult,
    node_id: int,
    tier: str | None = None,
  ) -> MiningNode | None:
    if node_id <= 0:
      return None
    for pool in (result.nodes, result.track_nodes or []):
      for node in pool:
        if node.node_id != node_id:
          continue
        if tier is not None and node.tier != tier:
          continue
        return node
    return None

  def _pivot_distance(self, result: NodeScanResult, x: float, y: float) -> float:
    return float(
      math.hypot(x - result.player_x, y - result.player_y)
    )

  def _can_snap_virtual_to(
    self,
    lock: TargetLock,
    result: NodeScanResult,
    x: float,
    y: float,
    *,
    from_track: bool,
    pool: list[MiningNode],
    live: MiningNode,
  ) -> bool:
    """Nunca colapsa o virtual em cima da seta do jogador."""
    dead = self.pivot_dead_zone_px
    pivot_dist = self._pivot_distance(result, x, y)
    if pivot_dist < 4.0:
      return False
    if pivot_dist <= dead:
      if not from_track or not lock.committed:
        return False
      if lock.anchor_offsets:
        if self._anchor_mismatch(live, pool, lock.anchor_offsets) > 20.0:
          return False
      if lock.locked_area > 0:
        ratio = live.area / lock.locked_area
        if ratio < 0.15 or ratio > 5.0:
          return False
      elif live.area > 40.0:
        return False
    return True

  def _remember_outside_virtual(self, lock: TargetLock, result: NodeScanResult) -> None:
    dead = self.pivot_dead_zone_px
    if self._pivot_distance(result, lock.virtual_x, lock.virtual_y) > dead + 2.0:
      lock.last_outside_x = lock.virtual_x
      lock.last_outside_y = lock.virtual_y

  def _advance_virtual_inertia(self, lock: TargetLock) -> None:
    lock.virtual_x += lock.vel_x
    lock.virtual_y += lock.vel_y
    lock.locked_x = lock.virtual_x
    lock.locked_y = lock.virtual_y

  def _recover_virtual_from_collapse(self, lock: TargetLock, result: NodeScanResult) -> None:
    """Se o virtual caiu na seta, restaura ao longo da ultima direcao conhecida."""
    pivot_dist = self._pivot_distance(result, lock.virtual_x, lock.virtual_y)
    if pivot_dist >= 5.0:
      return
    ref_x = lock.last_outside_x
    ref_y = lock.last_outside_y
    if ref_x == 0.0 and ref_y == 0.0 and (lock.pick_x or lock.pick_y):
      ref_x = lock.pick_x
      ref_y = lock.pick_y
    dx = ref_x - result.player_x
    dy = ref_y - result.player_y
    mag = math.hypot(dx, dy)
    if mag < 1.0:
      return
    hold = max(6.0, self.pivot_dead_zone_px + 3.0, lock.min_seen_distance_px)
    lock.virtual_x = result.player_x + hold * dx / mag
    lock.virtual_y = result.player_y + hold * dy / mag
    lock.locked_x = lock.virtual_x
    lock.locked_y = lock.virtual_y

  def _tier_candidates(
    self,
    result: NodeScanResult,
    lock: TargetLock,
  ) -> tuple[list[MiningNode], list[MiningNode]]:
    tier_all = [n for n in result.nodes if n.tier == lock.tier]
    pool = list(result.nodes)
    if result.track_nodes:
      for tracked in result.track_nodes:
        if tracked.tier != lock.tier:
          continue
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0 for node in tier_all
        ):
          tier_all.append(tracked)
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0
          for node in pool
          if node.tier == tracked.tier
        ):
          pool.append(tracked)
    dead = self.pivot_dead_zone_px
    if lock.approached_outside or lock.committed:
      return tier_all, pool
    outside = [n for n in tier_all if n.distance_px > dead]
    return (outside if outside else tier_all), pool

  def _reacquire_locked_node(
    self,
    result: NodeScanResult,
    lock: TargetLock,
    *,
    all_for_anchors: list[MiningNode] | None = None,
    close_hold_px: float = 34.0,
    close_stick_px: float = 10.0,
  ) -> MiningNode | None:
    """Re-encontra o blip travado entre os nos visiveis (mapa rola, coords mudam)."""
    candidates = [n for n in result.nodes if n.tier == lock.tier]
    if result.track_nodes:
      for tracked in result.track_nodes:
        if tracked.tier != lock.tier:
          continue
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0
          for node in candidates
        ):
          candidates.append(tracked)
    if not candidates:
      return None

    anchor_pool = all_for_anchors if all_for_anchors is not None else list(result.nodes)
    if result.track_nodes:
      for tracked in result.track_nodes:
        if tracked not in anchor_pool:
          anchor_pool = list(anchor_pool) + [tracked]

    if self._in_approach_zone(lock, close_hold_px=close_hold_px):
      return self._stick_to_lock(
        candidates,
        lock,
        close_stick_px=close_stick_px,
        close_hold_px=close_hold_px,
        all_for_anchors=anchor_pool,
      )

    dead = self.pivot_dead_zone_px
    if not lock.approached_outside and not lock.committed:
      far = [n for n in candidates if n.distance_px > dead]
      if far:
        candidates = far

    stuck = self._stick_to_lock(
      candidates,
      lock,
      close_stick_px=close_stick_px,
      close_hold_px=close_hold_px,
      all_for_anchors=anchor_pool,
    )
    if stuck is not None:
      return stuck

    if lock.anchor_offsets:
      ranked = sorted(
        candidates,
        key=lambda node: (
          self._anchor_mismatch(node, anchor_pool, lock.anchor_offsets),
          math.hypot(node.x - lock.locked_x, node.y - lock.locked_y),
        ),
      )
      if ranked and self._anchor_mismatch(ranked[0], anchor_pool, lock.anchor_offsets) <= 32.0:
        jump = math.hypot(ranked[0].x - lock.locked_x, ranked[0].y - lock.locked_y)
        if jump <= self._stick_radius(
          lock, close_stick_px=close_stick_px, close_hold_px=close_hold_px
        ):
          return ranked[0]

    nearest = min(
      candidates,
      key=lambda node: math.hypot(node.x - lock.locked_x, node.y - lock.locked_y),
    )
    jump = math.hypot(nearest.x - lock.locked_x, nearest.y - lock.locked_y)
    if jump <= self._stick_radius(
      lock, close_stick_px=close_stick_px, close_hold_px=close_hold_px
    ):
      return nearest
    return None

  def _center_exclusion_for_tier(self, tier: str) -> float:
    if tier == "gray":
      return self.center_exclusion_radius_px
    return min(self.center_exclusion_radius_px, self.colored_center_exclusion_px)

  def _build_anchors(
    self,
    locked: MiningNode,
    all_nodes: list[MiningNode],
    *,
    max_anchors: int = 10,
    max_radius_px: float = 130.0,
  ) -> tuple[tuple[float, float, str], ...]:
    scored: list[tuple[float, float, str, float]] = []
    for node in all_nodes:
      if math.hypot(node.x - locked.x, node.y - locked.y) < 3.0:
        continue
      dx = node.x - locked.x
      dy = node.y - locked.y
      dist = math.hypot(dx, dy)
      if dist < 6.0 or dist > max_radius_px:
        continue
      scored.append((dx, dy, node.tier, dist))
    scored.sort(key=lambda item: item[3])
    return tuple((dx, dy, tier) for dx, dy, tier, _ in scored[:max_anchors])

  def _anchor_mismatch(
    self,
    candidate: MiningNode,
    all_nodes: list[MiningNode],
    anchors: tuple[tuple[float, float, str], ...],
  ) -> float:
    if not anchors:
      return 0.0
    total = 0.0
    for ox, oy, tier in anchors:
      expected_x = candidate.x + ox
      expected_y = candidate.y + oy
      best = min(
        (
          math.hypot(node.x - expected_x, node.y - expected_y)
          for node in all_nodes
          if node.tier == tier
        ),
        default=999.0,
      )
      total += best if best < 999.0 else 40.0
    return total

  def _accept_locked_node(
    self,
    node: MiningNode,
    lock: TargetLock,
    all_nodes: list[MiningNode],
    *,
    allow_pivot_overlap: bool = False,
    anchor_limit: float = 45.0,
  ) -> bool:
    """Rejeita seta do jogador / blip dentro da zona morta do pivot."""
    if node.distance_px <= self.pivot_dead_zone_px:
      if lock.committed and lock.approached_outside:
        if lock.anchor_offsets:
          if self._anchor_mismatch(node, all_nodes, lock.anchor_offsets) > 28.0:
            return False
      elif not (allow_pivot_overlap and lock.approached_outside):
        return False

    if lock.locked_area > 0:
      ratio = node.area / lock.locked_area
      max_ratio = 6.5 if lock.committed else 4.5
      min_ratio = 0.12 if lock.committed else 0.2
      if ratio < min_ratio or ratio > max_ratio:
        return False
    elif node.area > 55.0:
      return False

    if lock.anchor_offsets:
      mismatch = self._anchor_mismatch(node, all_nodes, lock.anchor_offsets)
      if mismatch > anchor_limit:
        return False
    return True

  def _in_approach_zone(self, lock: TargetLock, *, close_hold_px: float) -> bool:
    """True quando ja entrou na reta final — nunca troca de blip."""
    return lock.committed or (
      lock.approached_outside
      and (
        lock.last_distance_px <= close_hold_px + 6.0
        or lock.min_seen_distance_px <= close_hold_px + 10.0
      )
    )

  def _stick_radius(
    self,
    lock: TargetLock,
    *,
    close_stick_px: float,
    close_hold_px: float,
  ) -> float:
    base = max(close_stick_px * 3.0, 28.0)
    if self._in_approach_zone(lock, close_hold_px=close_hold_px):
      base = max(base, 48.0)
    scaled = max(lock.last_distance_px, lock.pick_distance_px) * 0.38
    return min(max(base, scaled), 78.0)

  def _drift_position(
    self,
    lock: TargetLock,
    player_x: float,
    player_y: float,
  ) -> tuple[float, float]:
    if abs(lock.vel_x) + abs(lock.vel_y) > 0.35:
      return lock.locked_x + lock.vel_x, lock.locked_y + lock.vel_y
    dx = player_x - lock.locked_x
    dy = player_y - lock.locked_y
    dist = float(math.hypot(dx, dy))
    if dist < 0.4:
      return lock.locked_x, lock.locked_y
    step = max(1.6, min(7.5, lock.last_distance_px * 0.11, dist * 0.28))
    return lock.locked_x + (dx / dist) * step, lock.locked_y + (dy / dist) * step

  def _stick_to_lock(
    self,
    nodes: list[MiningNode],
    lock: TargetLock,
    *,
    close_stick_px: float,
    close_hold_px: float,
    all_for_anchors: list[MiningNode] | None = None,
  ) -> MiningNode | None:
    """Escolhe o candidato mais proximo da posicao travada (nao do pivot)."""
    if not nodes:
      return None
    stick = self._stick_radius(
      lock, close_stick_px=close_stick_px, close_hold_px=close_hold_px
    )
    near = [
      node
      for node in nodes
      if math.hypot(node.x - lock.locked_x, node.y - lock.locked_y) <= stick
    ]
    pool = near if near else list(nodes)

    def _score(node: MiningNode) -> tuple[float, float, float]:
      lock_err = math.hypot(node.x - lock.locked_x, node.y - lock.locked_y)
      pick_err = math.hypot(node.x - lock.pick_x, node.y - lock.pick_y)
      anchor = 0.0
      if lock.anchor_offsets and all_for_anchors:
        anchor = self._anchor_mismatch(node, all_for_anchors, lock.anchor_offsets)
      return (lock_err, anchor * 0.4, pick_err * 0.02)

    return min(pool, key=_score)

  def _can_overlap_recover(self, lock: TargetLock, *, close_hold_px: float) -> bool:
    return lock.approached_outside and (
      lock.min_seen_distance_px <= close_hold_px + 4.0 or lock.lost_frames > 0
    )

  def _recover_at_pivot(
    self,
    mask: np.ndarray,
    lock: TargetLock,
    *,
    player_x: float,
    player_y: float,
    search_radius_px: float = 26.0,
  ) -> MiningNode | None:
    """Detecta blip sob a seta (seta sobreposta ao no)."""
    if not lock.approached_outside:
      return None
    height, width = mask.shape[:2]
    x0 = max(0, int(player_x - search_radius_px))
    y0 = max(0, int(player_y - search_radius_px))
    x1 = min(width, int(player_x + search_radius_px) + 1)
    y1 = min(height, int(player_y + search_radius_px) + 1)
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0 or not np.any(crop):
      return None
    moments = cv2.moments(crop)
    blob_area = float(moments["m00"])
    if blob_area < 1.0:
      return None
    if lock.locked_area > 0:
      ratio = blob_area / lock.locked_area
      if ratio < 0.15 or ratio > 4.0:
        return None
    elif blob_area > 50.0:
      return None
    x = float(moments["m10"] / moments["m00"]) + x0
    y = float(moments["m01"] / moments["m00"]) + y0
    dist_px = float(math.hypot(x - player_x, y - player_y))
    return MiningNode(
      tier=lock.tier,
      x=x,
      y=y,
      radius=4.0,
      area=lock.locked_area or 12.0,
      distance_px=dist_px,
      circularity=1.0,
    )

  def _try_mask_recovery(
    self,
    result: NodeScanResult,
    lock: TargetLock,
    *,
    search_radius_px: float = 28.0,
    close_hold_px: float = 34.0,
  ) -> MiningNode | None:
    if not self._can_overlap_recover(lock, close_hold_px=close_hold_px):
      return None
    tier_mask = None
    if result.track_masks:
      tier_mask = result.track_masks.get(lock.tier)
    if tier_mask is None:
      tier_mask = result.masks.get(lock.tier)
    if tier_mask is None:
      return None
    at_pivot = self._recover_at_pivot(
      tier_mask,
      lock,
      player_x=result.player_x,
      player_y=result.player_y,
      search_radius_px=search_radius_px,
    )
    if at_pivot is not None:
      return at_pivot
    return self._recover_blip_from_mask(
      tier_mask,
      lock,
      player_x=result.player_x,
      player_y=result.player_y,
      search_radius_px=search_radius_px,
    )

  def _recover_blip_from_mask(
    self,
    mask: np.ndarray,
    lock: TargetLock,
    *,
    player_x: float,
    player_y: float,
    search_radius_px: float = 22.0,
  ) -> MiningNode | None:
    """Busca blob do tier travado perto da ultima posicao ou ao longo do raio pivot→alvo."""
    probes: list[tuple[float, float]] = [
      (player_x, player_y),
      (lock.locked_x, lock.locked_y),
    ]
    pred_x = lock.locked_x + lock.vel_x
    pred_y = lock.locked_y + lock.vel_y
    probes.append((pred_x, pred_y))

    dist = max(lock.last_distance_px, 2.0)
    bearing = math.radians(lock.last_bearing_deg)
    for scale in (1.0, 0.85, 0.7, 0.55, 1.15):
      probes.append(
        (
          player_x + math.cos(bearing) * dist * scale,
          player_y + math.sin(bearing) * dist * scale,
        )
      )

    height, width = mask.shape[:2]
    best_node: MiningNode | None = None
    best_err = 999.0
    for cx, cy in probes:
      x0 = max(0, int(cx - search_radius_px))
      y0 = max(0, int(cy - search_radius_px))
      x1 = min(width, int(cx + search_radius_px) + 1)
      y1 = min(height, int(cy + search_radius_px) + 1)
      crop = mask[y0:y1, x0:x1]
      if crop.size == 0 or not np.any(crop):
        continue
      moments = cv2.moments(crop)
      if moments["m00"] < 1.0:
        continue
      x = float(moments["m10"] / moments["m00"]) + x0
      y = float(moments["m01"] / moments["m00"]) + y0
      err = math.hypot(x - lock.locked_x, y - lock.locked_y)
      if err < best_err:
        best_err = err
        dist_px = float(math.hypot(x - player_x, y - player_y))
        best_node = MiningNode(
          tier=lock.tier,
          x=x,
          y=y,
          radius=4.0,
          area=lock.locked_area or 12.0,
          distance_px=dist_px,
          circularity=1.0,
        )
    return best_node

  def _scan_tier_unlocked(
    self,
    frame_bgr: np.ndarray,
    tier: str,
    *,
    player_x: float,
    player_y: float,
    player_exclusion_mask: np.ndarray | None = None,
  ) -> tuple[list[MiningNode], np.ndarray]:
    """Detecta tier travado sem mascara da seta (blip pode ficar sob o pivot)."""
    del player_exclusion_mask
    height, width = frame_bgr.shape[:2]
    minimap_mask = self._minimap_mask(width, height)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)

    raw = self._tier_mask(frame_hsv, tier)
    raw = cv2.bitwise_and(raw, minimap_mask)
    if tier == "gray":
      raw = cv2.bitwise_and(raw, cv2.bitwise_not(self._gray_road_exclusion(frame_bgr)))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    if tier != "gray":
      raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

    nodes = self._find_nodes_in_mask(
      raw,
      tier=tier,
      player_x=player_x,
      player_y=player_y,
      center_exclusion_radius_px=0.0,
      use_near_center_strict=False,
      gray=gray,
    )
    return nodes, raw

  def player_center(self, shape: tuple[int, ...]) -> tuple[float, float]:
    height, width = shape[:2]
    return width * self.player_center_ratio[0], height * self.player_center_ratio[1]

  def _minimap_mask(self, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cx = int(width * self.player_center_ratio[0])
    cy = int(height * self.player_center_ratio[1])
    radius = int(min(width, height) * (0.5 - self.minimap_circle_margin_ratio))
    cv2.circle(mask, (cx, cy), max(radius, 1), 255, -1)
    return mask

  def _road_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.inRange(gray, self.road_gray_min, self.road_gray_max)

  def _gray_road_exclusion(self, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Pavimento cinza claro (V até ~199) fora do núcleo branco do nó.

    road_gray_range sozinho para em ~150 e deixa faixa 150–199 — exatamente a
    que a máscara gray alargada come como “nó”.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    protect_lo = max(self.road_bright_protect_min, self.road_gray_max + 1)
    road_hi = max(self.road_gray_max, protect_lo - 1)
    road = cv2.inRange(gray, self.road_gray_min, road_hi)
    bright_core = cv2.inRange(gray, protect_lo, 255)
    protect = cv2.dilate(bright_core, np.ones((3, 3), np.uint8), iterations=1)
    excluded = cv2.bitwise_and(road, cv2.bitwise_not(protect))
    # Une fitas de estrada para não sobrarem blobs quase-circulares soltos.
    excluded = cv2.morphologyEx(
      excluded, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )
    return excluded

  def _tier_mask(self, frame_hsv: np.ndarray, tier: str) -> np.ndarray:
    spec = self.tier_colors_hsv.get(tier)
    if spec is None:
      return np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
    lower = np.array(spec["lower"], dtype=np.uint8)
    upper = np.array(spec["upper"], dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lower, upper)
    # Opt-in only: old full-H OR could widen S/V past calibrated gray and
    # reopen mid-gray roads. Calibrated tier_colors_hsv.gray is the default.
    if tier == "gray" and self.gray_achromatic_expand:
      v_lo = int(max(self.gray_achromatic_v_min, int(lower[2])))
      s_hi = int(min(self.gray_achromatic_s_max, int(upper[1])))
      achromatic = cv2.inRange(
        frame_hsv,
        np.array([0, 0, v_lo], dtype=np.uint8),
        np.array([179, s_hi, 255], dtype=np.uint8),
      )
      mask = cv2.bitwise_or(mask, achromatic)
    return mask

  @staticmethod
  def _disk_center(
    contour: np.ndarray,
    gray: np.ndarray | None = None,
  ) -> tuple[float, float, float]:
    """
    Centro do disco visual do nó.

    Moments do blob binário puxam para franja/sombra inferior (AA + crescent).
    minEnclosingCircle recupera o círculo; se houver gray, refina pelo núcleo
    claro (peso ~ brilho²) dentro desse círculo.
    """
    (ecx, ecy), er = cv2.minEnclosingCircle(contour)
    cx, cy, radius = float(ecx), float(ecy), float(er)
    if gray is None or radius < 1.5:
      return cx, cy, radius

    r_i = max(int(math.ceil(radius)) + 1, 2)
    ix, iy = int(round(cx)), int(round(cy))
    height, width = gray.shape[:2]
    x0 = max(0, ix - r_i)
    y0 = max(0, iy - r_i)
    x1 = min(width, ix + r_i + 1)
    y1 = min(height, iy + r_i + 1)
    if x1 <= x0 or y1 <= y0:
      return cx, cy, radius

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 0.75) ** 2
    bright = patch >= 150.0
    weights = np.where(inside & bright, np.maximum(patch - 120.0, 0.0) ** 2, 0.0)
    total = float(weights.sum())
    if total < 1.0:
      return cx, cy, radius
    return (
      float((xx * weights).sum() / total),
      float((yy * weights).sum() / total),
      radius,
    )

  def _find_nodes_in_mask(
    self,
    mask: np.ndarray,
    *,
    tier: str,
    player_x: float,
    player_y: float,
    center_exclusion_radius_px: float | None = None,
    use_near_center_strict: bool = True,
    gray: np.ndarray | None = None,
  ) -> list[MiningNode]:
    excl_radius = (
      center_exclusion_radius_px
      if center_exclusion_radius_px is not None
      else self.center_exclusion_radius_px
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nodes: list[MiningNode] = []
    for contour in contours:
      area = float(cv2.contourArea(contour))
      if area < self.min_blob_area or area > self.max_blob_area:
        continue
      x0, y0, bw, bh = cv2.boundingRect(contour)
      if bw < 1 or bh < 1:
        continue
      # Estradas / fitas: alongadas e pouco sólidas vs disco de nó.
      aspect = float(max(bw, bh)) / float(max(min(bw, bh), 1))
      if aspect > self.max_aspect_ratio:
        continue
      hull = cv2.convexHull(contour)
      hull_area = float(cv2.contourArea(hull))
      solidity = (area / hull_area) if hull_area > 1.0 else 0.0
      if solidity < self.min_solidity:
        continue
      perimeter = float(cv2.arcLength(contour, True))
      if perimeter < 1.0:
        continue
      circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
      min_circ = self.min_circularity
      # Disco: enclosing circle (+ núcleo claro) — evita bias Y da franja inferior.
      if circularity >= 0.45:
        x, y, enc_r = self._disk_center(contour, gray)
        radius = max(enc_r, math.sqrt(area / math.pi))
      else:
        filled = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(filled, [contour - np.array([[[x0, y0]]])], -1, 255, thickness=-1)
        moments = cv2.moments(filled, binaryImage=True)
        if moments["m00"] < 1.0:
          continue
        x = float(moments["m10"] / moments["m00"]) + float(x0)
        y = float(moments["m01"] / moments["m00"]) + float(y0)
        radius = math.sqrt(area / math.pi)
      if radius > self.max_enclosing_radius_px:
        continue
      dist = float(math.hypot(x - player_x, y - player_y))
      if dist < excl_radius:
        continue
      if use_near_center_strict and dist < self.near_center_strict_radius_px:
        min_circ = max(min_circ, self.near_center_min_circularity)
        if area > self.near_center_max_blob_area:
          continue
      if circularity < min_circ:
        continue
      nodes.append(
        MiningNode(
          tier=tier,
          x=x,
          y=y,
          radius=radius,
          area=area,
          distance_px=dist,
          circularity=circularity,
        )
      )
    return nodes

  def scan(
    self,
    frame_bgr: np.ndarray,
    *,
    player_exclusion_mask: np.ndarray | None = None,
    player_x: float | None = None,
    player_y: float | None = None,
  ) -> NodeScanResult:
    height, width = frame_bgr.shape[:2]
    nominal_x, nominal_y = self.player_center(frame_bgr.shape)
    origin_x = player_x if player_x is not None else nominal_x
    origin_y = player_y if player_y is not None else nominal_y
    minimap_mask = self._minimap_mask(width, height)
    gray_road_excl = self._gray_road_exclusion(frame_bgr)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    masks: dict[str, np.ndarray] = {}
    all_nodes: list[MiningNode] = []

    for tier in self.tier_colors_hsv:
      raw = self._tier_mask(frame_hsv, tier)
      raw = cv2.bitwise_and(raw, minimap_mask)
      if tier == "gray":
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(gray_road_excl))
      else:
        # Coloridos: só road_gray clássico (não estende até protect_min).
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(self._road_mask(frame_bgr)))
      if player_exclusion_mask is not None:
        raw = cv2.bitwise_and(raw, player_exclusion_mask)
      kernel = np.ones((2, 2), np.uint8)
      raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
      masks[tier] = raw
      if tier not in self.allowed_tiers:
        continue
      all_nodes.extend(
        self._find_nodes_in_mask(
          raw, tier=tier, player_x=origin_x, player_y=origin_y, gray=gray
        )
      )

    all_nodes.sort(key=lambda node: node.distance_px)
    target = self.nearest_node(
      all_nodes,
      min_distance_px=self.min_target_distance_px,
    )
    return NodeScanResult(
      nodes=all_nodes,
      target=target,
      masks=masks,
      player_x=origin_x,
      player_y=origin_y,
    )

  def _hough_bright_nodes(
    self,
    gray: np.ndarray,
    *,
    player_x: float,
    player_y: float,
    min_brightness: int,
    center_exclusion_radius_px: float,
    max_blob_area: int,
    usable_mask: np.ndarray | None = None,
  ) -> list[MiningNode]:
    """Fallback: circulos claros no minimapa (nos cinza perto da seta)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
      blur,
      cv2.HOUGH_GRADIENT,
      dp=1.2,
      minDist=max(int(center_exclusion_radius_px) + 6, 10),
      param1=90,
      param2=11,
      minRadius=2,
      maxRadius=14,
    )
    if circles is None:
      return []

    nodes: list[MiningNode] = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
      if cx < 0 or cy < 0 or cx >= gray.shape[1] or cy >= gray.shape[0]:
        continue
      if usable_mask is not None and usable_mask[cy, cx] == 0:
        continue
      if int(gray[cy, cx]) < min_brightness:
        continue
      dist = float(math.hypot(cx - player_x, cy - player_y))
      if dist < center_exclusion_radius_px:
        continue
      area = float(math.pi * radius * radius)
      if area < self.min_blob_area or area > max_blob_area:
        continue
      nodes.append(
        MiningNode(
          tier="gray",
          x=float(cx),
          y=float(cy),
          radius=float(radius),
          area=area,
          distance_px=dist,
          circularity=0.85,
        )
      )
    return nodes

  def scan_for_tile(
    self,
    frame_bgr: np.ndarray,
    *,
    player_exclusion_mask: np.ndarray | None = None,
    player_x: float | None = None,
    player_y: float | None = None,
    min_brightness: int = 135,
    center_exclusion_radius_px: float = 10.0,
    min_circularity: float = 0.35,
    max_blob_area: int = 120,
    merge_nearby_px: float = 10.0,
  ) -> NodeScanResult:
    """
    Deteccao permissiva para snapshot do tile.
    Nao remove blips em cima da estrada (problema principal do scan() normal).
    """
    height, width = frame_bgr.shape[:2]
    nominal_x, nominal_y = self.player_center(frame_bgr.shape)
    origin_x = player_x if player_x is not None else nominal_x
    origin_y = player_y if player_y is not None else nominal_y

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(gray, int(min_brightness), 255)
    low_sat = cv2.inRange(hsv, np.array([0, 0, min_brightness], dtype=np.uint8), np.array([179, 85, 255], dtype=np.uint8))
    mask = cv2.bitwise_and(bright, low_sat)

    for tier in self.allowed_tiers:
      tier_mask = self._tier_mask(hsv, tier)
      mask = cv2.bitwise_or(mask, tier_mask)

    usable = np.full((height, width), 255, dtype=np.uint8)
    cx, cy = int(round(origin_x)), int(round(origin_y))
    cv2.circle(usable, (cx, cy), int(center_exclusion_radius_px), 0, -1)
    mask = cv2.bitwise_and(mask, usable)
    if player_exclusion_mask is not None:
      mask = cv2.bitwise_and(mask, player_exclusion_mask)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    nodes = self._find_nodes_in_mask(
      mask,
      tier="gray",
      player_x=origin_x,
      player_y=origin_y,
      center_exclusion_radius_px=center_exclusion_radius_px,
      use_near_center_strict=False,
      gray=gray,
    )
    nodes = [
      node for node in nodes
      if node.area <= max_blob_area and node.circularity >= min_circularity
      and node.distance_px >= center_exclusion_radius_px
    ]
    nodes.extend(
      self._hough_bright_nodes(
        gray,
        player_x=origin_x,
        player_y=origin_y,
        min_brightness=max(min_brightness - 15, 110),
        center_exclusion_radius_px=center_exclusion_radius_px,
        max_blob_area=max_blob_area,
        usable_mask=usable,
      )
    )
    nodes.sort(key=lambda node: node.distance_px)

    deduped: list[MiningNode] = []
    for node in nodes:
      if any(
        math.hypot(node.x - kept.x, node.y - kept.y) < merge_nearby_px
        for kept in deduped
      ):
        continue
      deduped.append(node)

    return NodeScanResult(
      nodes=deduped,
      target=self.nearest_node(deduped, min_distance_px=0.0),
      masks={"tile": mask},
      player_x=origin_x,
      player_y=origin_y,
    )

  def scan_blips(
    self,
    frame_bgr: np.ndarray,
    *,
    player_exclusion_mask: np.ndarray | None = None,
    player_x: float | None = None,
    player_y: float | None = None,
    min_distance_px: float = 18.0,
    track_lock: TargetLock | None = None,
  ) -> NodeScanResult:
    """
    Deteccao permissiva para bot ao vivo.
    Nos coloridos (laranja/ciano) NAO sao filtrados pela mascara de estrada.
    """
    height, width = frame_bgr.shape[:2]
    nominal_x, nominal_y = self.player_center(frame_bgr.shape)
    origin_x = player_x if player_x is not None else nominal_x
    origin_y = player_y if player_y is not None else nominal_y
    minimap_mask = self._minimap_mask(width, height)
    gray_road_excl = self._gray_road_exclusion(frame_bgr)
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    masks: dict[str, np.ndarray] = {}
    all_nodes: list[MiningNode] = []
    kernel = np.ones((2, 2), np.uint8)
    # Com lock ativo: permite ver o nó perto do pivot (chegada / minerar).
    locked_close = track_lock is not None

    for tier in self.tier_colors_hsv:
      if tier not in self.allowed_tiers:
        continue
      raw = self._tier_mask(frame_hsv, tier)
      raw = cv2.bitwise_and(raw, minimap_mask)
      if tier == "gray":
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(gray_road_excl))
      if player_exclusion_mask is not None:
        raw = cv2.bitwise_and(raw, player_exclusion_mask)
      raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
      # CLOSE une fitas de estrada em blobs falsos — só em nós coloridos.
      if tier != "gray":
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
      masks[tier] = raw
      excl = self._center_exclusion_for_tier(tier)
      if locked_close:
        excl = min(excl, 8.0)
      all_nodes.extend(
        self._find_nodes_in_mask(
          raw,
          tier=tier,
          player_x=origin_x,
          player_y=origin_y,
          center_exclusion_radius_px=excl,
          use_near_center_strict=(tier == "gray" and not locked_close),
          gray=gray,
        )
      )

    if not all_nodes:
      tile_scan = self.scan_for_tile(
        frame_bgr,
        player_exclusion_mask=player_exclusion_mask,
        player_x=origin_x,
        player_y=origin_y,
        min_brightness=185,
        center_exclusion_radius_px=max(self.center_exclusion_radius_px * 0.35, 12.0),
        min_circularity=max(0.55, self.min_circularity - 0.08),
        max_blob_area=self.max_blob_area,
      )
      all_nodes = tile_scan.nodes

    all_nodes.sort(key=lambda node: node.distance_px)
    target = self.nearest_node(all_nodes, min_distance_px=min_distance_px)

    track_nodes: list[MiningNode] | None = None
    track_masks: dict[str, np.ndarray] | None = None
    if track_lock is not None:
      track_nodes, track_mask = self._scan_tier_unlocked(
        frame_bgr,
        track_lock.tier,
        player_x=origin_x,
        player_y=origin_y,
        player_exclusion_mask=player_exclusion_mask,
      )
      track_masks = {track_lock.tier: track_mask}

    return self._tag_scan_nodes(
      NodeScanResult(
        nodes=all_nodes,
        target=target,
        masks=masks,
        player_x=origin_x,
        player_y=origin_y,
        track_nodes=track_nodes,
        track_masks=track_masks,
      )
    )

  def nearest_node(
    self,
    nodes: list[MiningNode],
    *,
    tier: str | None = None,
    min_distance_px: float = 0.0,
    exclude_xy: tuple[float, float] | None = None,
    exclude_points: list[tuple[float, float]] | None = None,
    exclude_radius_px: float = 35.0,
  ) -> MiningNode | None:
    excludes = list(exclude_points or [])
    if exclude_xy is not None:
      excludes.append(exclude_xy)
    best = None
    for node in nodes:
      if tier is not None and node.tier != tier:
        continue
      if node.distance_px < min_distance_px:
        continue
      blocked = False
      for ex, ey in excludes:
        if math.hypot(node.x - ex, node.y - ey) < exclude_radius_px:
          blocked = True
          break
      if blocked:
        continue
      if best is None or node.distance_px < best.distance_px:
        best = node
    return best

  def make_target_lock(self, result: NodeScanResult, node: MiningNode) -> TargetLock:
    bearing = bearing_deg(result.player_x, result.player_y, node.x, node.y)
    anchor_pool = list(result.nodes)
    if result.track_nodes:
      for tracked in result.track_nodes:
        if tracked.tier != node.tier:
          continue
        if not any(
          math.hypot(tracked.x - n.x, tracked.y - n.y) < 7.0 for n in anchor_pool
        ):
          anchor_pool.append(tracked)
    anchors = self._build_anchors(node, anchor_pool)
    dead = self.pivot_dead_zone_px
    lock = TargetLock(
      tier=node.tier,
      node_id=int(node.node_id or 0),
      locked_x=node.x,
      locked_y=node.y,
      pick_distance_px=node.distance_px,
      last_distance_px=node.distance_px,
      locked_area=node.area,
      last_bearing_deg=bearing,
      lost_frames=0,
      min_seen_distance_px=node.distance_px,
      approached_outside=node.distance_px > dead,
      pick_x=node.x,
      pick_y=node.y,
      virtual_x=node.x,
      virtual_y=node.y,
      last_outside_x=node.x if node.distance_px > dead + 2.0 else 0.0,
      last_outside_y=node.y if node.distance_px > dead + 2.0 else 0.0,
      anchor_offsets=anchors,
    )
    if node.node_id:
      self.node_registry.pin(int(node.node_id))
    return lock

  def establish_lock(
    self,
    frame_bgr: np.ndarray,
    scan: NodeScanResult,
    node: MiningNode,
    *,
    player_exclusion_mask: np.ndarray | None = None,
    player_x: float,
    player_y: float,
    min_distance_px: float = 18.0,
  ) -> TargetLock:
    """Trava no blip escolhido — coords do scan ao vivo (sem rescan que desloca)."""
    del frame_bgr, player_exclusion_mask, player_x, player_y, min_distance_px
    pick = node
    if scan.track_nodes:
      near = [
        n
        for n in scan.track_nodes
        if n.tier == node.tier
        and math.hypot(n.x - node.x, n.y - node.y) <= 14.0
      ]
      if near:
        pick = min(
          near,
          key=lambda candidate: math.hypot(candidate.x - node.x, candidate.y - node.y),
        )
    return self.make_target_lock(scan, pick)

  def track_target(
    self,
    result: NodeScanResult,
    lock: TargetLock,
    *,
    chain_radius_px: float = 22.0,
    max_lost_frames: int = 10,
    close_hold_px: float = 34.0,
    close_coast_px: float = 12.0,
    close_stick_px: float = 10.0,
  ) -> tuple[MiningNode | None, TargetLock]:
    """Mantem o mesmo blip — coords sempre do no detectado na tela (sem drift)."""
    del chain_radius_px, max_lost_frames, close_coast_px

    tier_nodes, all_for_anchors = self._tier_candidates(result, lock)
    committed = (
      lock.committed
      or (
        lock.approached_outside
        and lock.last_distance_px <= close_hold_px
      )
    )
    in_approach = self._in_approach_zone(lock, close_hold_px=close_hold_px)
    allow_overlap = committed or in_approach or lock.approached_outside

    def _with_pivot_dist(node: MiningNode, *, ghost: bool = False) -> MiningNode:
      dist = float(
        math.hypot(node.x - result.player_x, node.y - result.player_y)
      )
      return MiningNode(
        tier=node.tier,
        x=node.x,
        y=node.y,
        radius=node.radius,
        area=node.area,
        distance_px=dist,
        circularity=node.circularity,
        ghost=ghost,
      )

    def _hold_last() -> tuple[MiningNode | None, TargetLock]:
      lock.lost_frames += 1
      dist = float(
        math.hypot(lock.locked_x - result.player_x, lock.locked_y - result.player_y)
      )
      if dist <= self.pivot_dead_zone_px and not lock.approached_outside:
        return None, lock
      ghost = _with_pivot_dist(
        MiningNode(
          tier=lock.tier,
          x=lock.locked_x,
          y=lock.locked_y,
          radius=4.0,
          area=lock.locked_area or 12.0,
          distance_px=dist,
          circularity=1.0,
        ),
        ghost=True,
      )
      lock.last_distance_px = dist
      return ghost, lock

    def _is_live(node: MiningNode) -> bool:
      for candidate in tier_nodes:
        if math.hypot(candidate.x - node.x, candidate.y - node.y) < 6.0:
          return True
      return False

    def _finalize(best: MiningNode) -> tuple[MiningNode | None, TargetLock]:
      live = _is_live(best)
      accepted = self._accept_locked_node(
        best, lock, all_for_anchors, allow_pivot_overlap=allow_overlap
      )
      if not accepted and not live:
        return _hold_last()
      if not accepted and live and best.distance_px <= self.pivot_dead_zone_px:
        if not lock.approached_outside and lock.pick_distance_px > self.pivot_dead_zone_px + 6.0:
          return _hold_last()

      best = _with_pivot_dist(best)
      bearing = bearing_deg(result.player_x, result.player_y, best.x, best.y)
      vel_x = 0.5 * (best.x - lock.locked_x) + 0.5 * lock.vel_x
      vel_y = 0.5 * (best.y - lock.locked_y) + 0.5 * lock.vel_y
      min_seen = lock.min_seen_distance_px
      if best.distance_px > self.pivot_dead_zone_px:
        min_seen = min(lock.min_seen_distance_px, best.distance_px)
      approached = lock.approached_outside or best.distance_px > self.pivot_dead_zone_px
      now_committed = (
        (lock.committed or (approached and best.distance_px <= close_hold_px))
        and approached
        and min_seen <= close_hold_px + 6.0
      )
      new_lock = TargetLock(
        tier=lock.tier,
        locked_x=best.x,
        locked_y=best.y,
        pick_distance_px=lock.pick_distance_px,
        last_distance_px=best.distance_px,
        locked_area=lock.locked_area or best.area,
        last_bearing_deg=bearing,
        vel_x=vel_x,
        vel_y=vel_y,
        lost_frames=0,
        committed=now_committed,
        min_seen_distance_px=min_seen,
        approached_outside=approached,
        pick_x=lock.pick_x,
        pick_y=lock.pick_y,
        anchor_offsets=lock.anchor_offsets,
      )
      return best, new_lock

    def _pick_live() -> MiningNode | None:
      if not tier_nodes:
        return None
      pred_x = lock.locked_x + lock.vel_x
      pred_y = lock.locked_y + lock.vel_y
      gate_px = 30.0 + lock.lost_frames * 7.0
      gate_px = max(gate_px, min(lock.last_distance_px * 0.45, 72.0))

      def _score(node: MiningNode) -> tuple[float, float, float]:
        screen_err = math.hypot(node.x - pred_x, node.y - pred_y)
        anchor = (
          self._anchor_mismatch(node, all_for_anchors, lock.anchor_offsets)
          if lock.anchor_offsets
          else 0.0
        )
        area_err = abs(node.area - lock.locked_area) if lock.locked_area > 0 else 0.0
        penalty = 0.0 if screen_err <= gate_px else screen_err
        jump = math.hypot(node.x - lock.locked_x, node.y - lock.locked_y)
        if lock.pick_distance_px >= 18.0 and jump > 32.0:
          penalty += jump * 0.85
        if not allow_overlap and node.distance_px <= self.pivot_dead_zone_px:
          penalty += 120.0
        return (penalty, anchor * 0.35, area_err * 0.02)

      pool = tier_nodes
      if in_approach or committed:
        stuck = self._stick_to_lock(
          tier_nodes,
          lock,
          close_stick_px=close_stick_px,
          close_hold_px=close_hold_px,
          all_for_anchors=all_for_anchors,
        )
        if stuck is not None:
          pool = [stuck]
      return min(pool, key=_score)

    live = _pick_live()
    if live is not None:
      jump_px = math.hypot(live.x - lock.locked_x, live.y - lock.locked_y)
      max_jump = max(36.0, lock.last_distance_px * 0.55) + lock.lost_frames * 6.0
      if in_approach and jump_px > max_jump:
        stuck = self._stick_to_lock(
          tier_nodes,
          lock,
          close_stick_px=close_stick_px,
          close_hold_px=close_hold_px,
          all_for_anchors=all_for_anchors,
        )
        if stuck is not None:
          live = stuck
      return _finalize(live)

    reacquired = self._reacquire_locked_node(
      result,
      lock,
      all_for_anchors=all_for_anchors,
      close_hold_px=close_hold_px,
      close_stick_px=close_stick_px,
    )
    if reacquired is not None:
      return _finalize(reacquired)

    if tier_nodes:
      snap = min(
        tier_nodes,
        key=lambda node: math.hypot(node.x - lock.locked_x, node.y - lock.locked_y),
      )
      return _finalize(snap)

    return _hold_last()

  def track_by_id(
    self,
    result: NodeScanResult,
    lock: TargetLock,
    *,
    close_hold_px: float = 34.0,
    peek_radius_px: float = 28.0,
  ) -> tuple[MiningNode | None, TargetLock]:
    """Trava por node_id + no virtual (inercia) — nunca troca ate E/F8."""
    if lock.node_id <= 0:
      return self.track_virtual(result, lock, close_hold_px=close_hold_px)

    if lock.virtual_x == 0.0 and lock.virtual_y == 0.0:
      lock.virtual_x = lock.pick_x or lock.locked_x
      lock.virtual_y = lock.pick_y or lock.locked_y

    pool = self._anchor_pool(result)
    pred_x = lock.virtual_x + lock.vel_x
    pred_y = lock.virtual_y + lock.vel_y
    near_pivot = lock.committed or lock.min_seen_distance_px <= close_hold_px + 8.0

    def _valid_live(
      candidate: MiningNode | None,
      *,
      from_track: bool,
    ) -> MiningNode | None:
      if candidate is None:
        return None
      if candidate.node_id != lock.node_id or candidate.tier != lock.tier:
        return None
      if lock.anchor_offsets:
        if self._anchor_mismatch(candidate, pool, lock.anchor_offsets) > 24.0:
          return None
      pivot_dist = candidate.distance_px
      if pivot_dist <= self.pivot_dead_zone_px and not from_track:
        return None
      if pivot_dist <= self.pivot_dead_zone_px and from_track:
        if not lock.committed:
          return None
        if lock.anchor_offsets:
          if self._anchor_mismatch(candidate, pool, lock.anchor_offsets) > 20.0:
            return None
        if lock.locked_area > 0:
          ratio = candidate.area / lock.locked_area
          if ratio < 0.15 or ratio > 5.0:
            return None
      elif not self._accept_locked_node(
        candidate, lock, pool, anchor_limit=34.0, allow_pivot_overlap=from_track
      ):
        return None
      return candidate

    live: MiningNode | None = None
    live_from_track = False
    search_order: list[tuple[list[MiningNode] | None, bool]] = []
    if near_pivot and result.track_nodes:
      search_order.append((result.track_nodes, True))
    search_order.append((result.nodes, False))
    if result.track_nodes and not near_pivot:
      search_order.append((result.track_nodes, True))

    for source, from_track in search_order:
      if not source:
        continue
      exact: MiningNode | None = None
      for candidate in source:
        if candidate.node_id == lock.node_id and candidate.tier == lock.tier:
          exact = candidate
          break
      if exact is not None:
        exact = _valid_live(exact, from_track=from_track)
      if exact is None:
        for candidate in source:
          if candidate.node_id != lock.node_id or candidate.tier != lock.tier:
            continue
          if math.hypot(candidate.x - pred_x, candidate.y - pred_y) > peek_radius_px:
            continue
          exact = _valid_live(candidate, from_track=from_track)
          if exact is not None:
            break
      if exact is not None:
        live = exact
        live_from_track = from_track
        break

    using_virtual = False
    if live is not None:
      obs_dx = live.x - lock.virtual_x
      obs_dy = live.y - lock.virtual_y
      lock.vel_x = 0.55 * obs_dx + 0.45 * lock.vel_x
      lock.vel_y = 0.55 * obs_dy + 0.45 * lock.vel_y
      if self._can_snap_virtual_to(
        lock,
        result,
        live.x,
        live.y,
        from_track=live_from_track,
        pool=pool,
        live=live,
      ):
        lock.virtual_x = live.x
        lock.virtual_y = live.y
        lock.locked_x = live.x
        lock.locked_y = live.y
        lock.locked_area = live.area
        lock.lost_frames = 0
      else:
        self._advance_virtual_inertia(lock)
        using_virtual = True
        lock.lost_frames += 1
    else:
      self._advance_virtual_inertia(lock)
      lock.lost_frames += 1
      using_virtual = True
      if lock.lost_frames > 3:
        lock.vel_x *= 0.88
        lock.vel_y *= 0.88

    self._recover_virtual_from_collapse(lock, result)
    self._remember_outside_virtual(lock, result)

    dist = self._pivot_distance(result, lock.virtual_x, lock.virtual_y)
    dead = self.pivot_dead_zone_px
    if dist < lock.min_seen_distance_px:
      lock.min_seen_distance_px = dist
    approached = lock.approached_outside or dist > dead
    if approached and dist <= close_hold_px:
      lock.committed = True
    lock.approached_outside = approached
    lock.last_distance_px = dist
    lock.last_bearing_deg = bearing_deg(
      result.player_x,
      result.player_y,
      lock.virtual_x,
      lock.virtual_y,
    )

    if dist <= dead and not lock.approached_outside and not lock.committed:
      return None, lock

    node = MiningNode(
      tier=lock.tier,
      x=lock.virtual_x,
      y=lock.virtual_y,
      radius=4.0,
      area=lock.locked_area or 12.0,
      distance_px=dist,
      circularity=1.0,
      virtual=using_virtual or live is None,
      ghost=using_virtual and lock.lost_frames > 6,
      node_id=lock.node_id,
    )
    return node, lock

  def track_virtual(
    self,
    result: NodeScanResult,
    lock: TargetLock,
    *,
    close_hold_px: float = 34.0,
    peek_radius_px: float = 30.0,
  ) -> tuple[MiningNode | None, TargetLock]:
    """
    Alvo virtual fixado no centro do no ao travar.
    Move por inercia (vel); peek so calibra velocidade — nunca salta pra outro blip.
    """
    if lock.virtual_x == 0.0 and lock.virtual_y == 0.0:
      lock.virtual_x = lock.pick_x or lock.locked_x
      lock.virtual_y = lock.pick_y or lock.locked_y

    pred_x = lock.virtual_x + lock.vel_x
    pred_y = lock.virtual_y + lock.vel_y

    pool = list(result.nodes)
    track_set = list(result.track_nodes or [])
    if track_set:
      for tracked in track_set:
        if tracked not in pool:
          pool.append(tracked)

    tier_nodes = [n for n in result.nodes if n.tier == lock.tier]
    near_pivot = lock.committed or lock.min_seen_distance_px <= close_hold_px + 8.0
    if near_pivot and track_set:
      for tracked in track_set:
        if tracked.tier != lock.tier:
          continue
        if lock.node_id > 0 and tracked.node_id != lock.node_id:
          continue
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0
          for node in tier_nodes
        ):
          tier_nodes.insert(0, tracked)
    elif track_set:
      for tracked in track_set:
        if tracked.tier != lock.tier:
          continue
        if lock.node_id > 0 and tracked.node_id != lock.node_id:
          continue
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0
          for node in tier_nodes
        ):
          tier_nodes.append(tracked)

    peek = None
    peek_from_track = False
    if tier_nodes:
      near = [
        n
        for n in tier_nodes
        if math.hypot(n.x - pred_x, n.y - pred_y) <= peek_radius_px
        and (lock.node_id <= 0 or n.node_id == lock.node_id)
      ]
      if near and lock.anchor_offsets:
        scored = [
          (n, self._anchor_mismatch(n, pool, lock.anchor_offsets))
          for n in near
        ]
        best_mismatch = min(score for _, score in scored)
        candidates = [
          n
          for n, score in scored
          if score <= min(22.0, best_mismatch + 3.0)
        ]
        if candidates:
          peek = min(
            candidates,
            key=lambda node: math.hypot(node.x - pred_x, node.y - pred_y),
          )
      elif near:
        peek = min(
          near,
          key=lambda node: math.hypot(node.x - pred_x, node.y - pred_y),
        )
      if peek is not None and track_set:
        peek_from_track = any(
          math.hypot(peek.x - t.x, peek.y - t.y) < 5.0 for t in track_set
        )

    if peek is not None:
      obs_dx = peek.x - lock.virtual_x
      obs_dy = peek.y - lock.virtual_y
      obs_jump = float(math.hypot(obs_dx, obs_dy))
      pivot_ok = self._can_snap_virtual_to(
        lock,
        result,
        peek.x,
        peek.y,
        from_track=peek_from_track,
        pool=pool,
        live=peek,
      )
      if obs_jump <= max(22.0, peek_radius_px * 0.75) and (
        pivot_ok or peek.distance_px > self.pivot_dead_zone_px
      ):
        lock.vel_x = 0.55 * obs_dx + 0.45 * lock.vel_x
        lock.vel_y = 0.55 * obs_dy + 0.45 * lock.vel_y
        if peek.area > 0:
          lock.locked_area = peek.area
        lock.lost_frames = 0

    self._advance_virtual_inertia(lock)
    if peek is None:
      lock.lost_frames += 1
      if lock.lost_frames > 4:
        lock.vel_x *= 0.9
        lock.vel_y *= 0.9

    self._recover_virtual_from_collapse(lock, result)
    self._remember_outside_virtual(lock, result)

    dist = float(
      math.hypot(
        lock.virtual_x - result.player_x,
        lock.virtual_y - result.player_y,
      )
    )
    dead = self.pivot_dead_zone_px
    if dist < lock.min_seen_distance_px:
      lock.min_seen_distance_px = dist
    approached = lock.approached_outside or dist > dead
    if approached and dist <= close_hold_px:
      lock.committed = True
    lock.approached_outside = approached
    lock.last_distance_px = dist
    lock.locked_x = lock.virtual_x
    lock.locked_y = lock.virtual_y
    lock.last_bearing_deg = bearing_deg(
      result.player_x,
      result.player_y,
      lock.virtual_x,
      lock.virtual_y,
    )

    if dist <= dead and not lock.approached_outside and not lock.committed:
      return None, lock

    node = MiningNode(
      tier=lock.tier,
      x=lock.virtual_x,
      y=lock.virtual_y,
      radius=4.0,
      area=lock.locked_area or 12.0,
      distance_px=dist,
      circularity=1.0,
      virtual=True,
      ghost=lock.lost_frames > 6,
      node_id=lock.node_id or None,
    )
    return node, lock

  def _ghost_from_lock(
    self, lock: TargetLock, result: NodeScanResult
  ) -> tuple[MiningNode | None, TargetLock]:
    lock.lost_frames += 1
    dist = float(
      math.hypot(lock.locked_x - result.player_x, lock.locked_y - result.player_y)
    )
    if dist <= self.pivot_dead_zone_px and not lock.approached_outside:
      return None, lock
    ghost = MiningNode(
      tier=lock.tier,
      x=lock.locked_x,
      y=lock.locked_y,
      radius=4.0,
      area=lock.locked_area or 12.0,
      distance_px=dist,
      circularity=1.0,
      ghost=True,
    )
    lock.last_distance_px = dist
    return ghost, lock

  def pick_next_after_switch(
    self,
    result: NodeScanResult,
    lock: TargetLock | None,
    *,
    min_distance_px: float = 0.0,
  ) -> MiningNode | None:
    """Proximo alvo ao pressionar E/F8 — pula para outro node_id."""
    pool = list(result.nodes)
    if lock is not None and result.track_nodes:
      for tracked in result.track_nodes:
        if lock is not None and tracked.tier != lock.tier:
          continue
        if not any(
          math.hypot(tracked.x - node.x, tracked.y - node.y) < 7.0 for node in pool
        ):
          pool.append(tracked)
    nodes = sorted(pool, key=lambda node: node.distance_px)
    eligible = [n for n in nodes if n.distance_px >= min_distance_px]
    if not eligible:
      return None
    if lock is None or lock.node_id <= 0:
      return eligible[0]

    others = [
      n
      for n in eligible
      if n.node_id is not None and n.node_id != lock.node_id
    ]
    if others:
      return others[0]

    for index, node in enumerate(eligible):
      if node.tier != lock.tier:
        continue
      if node.node_id == lock.node_id:
        continue
      return node
    return eligible[0] if eligible else None

  def node_from_lock(
    self,
    lock: TargetLock,
    *,
    player_x: float,
    player_y: float,
  ) -> MiningNode:
    dist = float(math.hypot(lock.locked_x - player_x, lock.locked_y - player_y))
    return MiningNode(
      tier=lock.tier,
      x=lock.locked_x,
      y=lock.locked_y,
      radius=4.0,
      area=12.0,
      distance_px=dist,
      circularity=1.0,
      ghost=True,
    )

  def resolve_locked_target(
    self,
    result: NodeScanResult,
    locked: MiningNode | None,
    *,
    lock_radius_px: float = 32.0,
  ) -> MiningNode | None:
    """Compatibilidade com chamadas antigas."""
    if locked is None:
      return result.target
    lock = TargetLock(
      tier=locked.tier,
      locked_x=locked.x,
      locked_y=locked.y,
      pick_distance_px=locked.distance_px,
      last_distance_px=locked.distance_px,
    )
    node, _ = self.track_target(result, lock, chain_radius_px=lock_radius_px)
    return node

  def is_near_target(self, target: MiningNode | None, *, arrive_px: float) -> bool:
    if target is None:
      return False
    return target.distance_px <= arrive_px

  def debug_frame(
    self,
    frame_bgr: np.ndarray,
    result: NodeScanResult,
    *,
    allowed_only: bool = True,
    locked_target: MiningNode | None = None,
    arrow_tip_x: float | None = None,
    arrow_tip_y: float | None = None,
    heading_error_deg: float | None = None,
  ) -> np.ndarray:
    debug = frame_bgr.copy()
    px, py = int(result.player_x), int(result.player_y)

    if locked_target is None:
      for node in result.nodes:
        cv2.circle(debug, (int(node.x), int(node.y)), 2, (120, 120, 120), 1)
      if result.target is not None:
        tx, ty = int(result.target.x), int(result.target.y)
        cv2.circle(debug, (tx, ty), 5, (0, 220, 255), 1)
        cv2.putText(
          debug,
          f"proximo {int(result.target.distance_px)}px",
          (tx + 6, ty - 6),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.38,
          (0, 220, 255),
          1,
          cv2.LINE_AA,
        )
    else:
      tx, ty = int(locked_target.x), int(locked_target.y)
      if arrow_tip_x is not None and arrow_tip_y is not None:
        ox, oy = int(arrow_tip_x), int(arrow_tip_y)
      else:
        ox, oy = px, py
      cv2.line(debug, (ox, oy), (tx, ty), (255, 0, 255), 2)
      cv2.circle(debug, (tx, ty), 8, (255, 0, 255), 2)
      cv2.putText(
        debug,
        f"ALVO {int(locked_target.distance_px)}px",
        (tx + 8, max(ty - 8, 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
      )

    if arrow_tip_x is not None and arrow_tip_y is not None:
      tip_x, tip_y = int(arrow_tip_x), int(arrow_tip_y)
      cv2.line(debug, (px, py), (tip_x, tip_y), (0, 255, 255), 2)
      cv2.circle(debug, (tip_x, tip_y), 4, (0, 255, 255), -1)
      cv2.putText(
        debug,
        "PONTA",
        (tip_x + 4, tip_y + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
      )

    cv2.drawMarker(debug, (px, py), (80, 80, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    if heading_error_deg is not None:
      cv2.putText(
        debug,
        f"rumo {heading_error_deg:.0f}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
      )
    return debug
