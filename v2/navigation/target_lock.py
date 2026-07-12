"""
Trava e rastreia blip alvo — só coordenadas de tela.
"""

from __future__ import annotations

import math

from v2.core.types import Blip, TargetLock


class TargetLocker:
  def __init__(
    self,
    *,
    min_pick_px: float = 18.0,
    max_lost_frames: int = 15,
    max_lost_frames_close: int = 45,
    track_radius_px: float = 42.0,
    close_hold_px: float = 38.0,
    close_track_radius_px: float = 22.0,
    pin_dist_px: float = 28.0,
    done_radius_px: float = 28.0,
    max_regress_px: float = 12.0,
  ) -> None:
    self.min_pick_px = min_pick_px
    self.max_lost_frames = max_lost_frames
    self.max_lost_frames_close = max_lost_frames_close
    self.track_radius_px = track_radius_px
    self.close_hold_px = close_hold_px
    self.close_track_radius_px = close_track_radius_px
    self.pin_dist_px = pin_dist_px
    self.done_radius_px = done_radius_px
    self.max_regress_px = max_regress_px
    self._lock: TargetLock | None = None
    self._next_id = 1
    self._done_points: list[tuple[float, float]] = []
    self._min_dist_seen: float | None = None

  @property
  def lock(self) -> TargetLock | None:
    return self._lock

  def unlock(self) -> None:
    self._lock = None
    self._min_dist_seen = None

  def mark_done(self) -> None:
    if self._lock is not None:
      self._done_points.append((self._lock.x, self._lock.y))
    self.unlock()

  def _dist_to_pivot(self, x: float, y: float, pivot: tuple[float, float]) -> float:
    return float(math.hypot(x - pivot[0], y - pivot[1]))

  def _update_min_dist(self, pivot: tuple[float, float]) -> float | None:
    if self._lock is None:
      return None
    dist = self._dist_to_pivot(self._lock.x, self._lock.y, pivot)
    if self._min_dist_seen is None:
      self._min_dist_seen = dist
    else:
      self._min_dist_seen = min(self._min_dist_seen, dist)
    return dist

  def _is_done_area(self, x: float, y: float) -> bool:
    for dx, dy in self._done_points:
      if math.hypot(x - dx, y - dy) < self.done_radius_px:
        return True
    return False

  def _pick_blip(
    self,
    blips: tuple[Blip, ...],
    pivot: tuple[float, float],
    *,
    min_dist: float | None = None,
  ) -> Blip | None:
    px, py = pivot
    pick_min = self.min_pick_px if min_dist is None else min_dist
    ranked = sorted(
      blips,
      key=lambda b: math.hypot(b.x - px, b.y - py),
    )
    for blip in ranked:
      dist = math.hypot(blip.x - px, blip.y - py)
      if dist < pick_min:
        continue
      if self._is_done_area(blip.x, blip.y):
        continue
      return blip
    return None

  def lock_nearest(self, blips: tuple[Blip, ...], pivot: tuple[float, float]) -> TargetLock | None:
    best = self._pick_blip(blips, pivot)
    if best is None:
      return None
    self._lock = TargetLock(
      track_id=self._next_id,
      x=best.x,
      y=best.y,
      tier=best.tier,
      lost_frames=0,
    )
    self._next_id += 1
    self._min_dist_seen = self._dist_to_pivot(best.x, best.y, pivot)
    return self._lock

  def _accept_blip(
    self,
    blip: Blip,
    pivot: tuple[float, float],
    *,
    cur_dist: float,
  ) -> bool:
    if self._lock is None:
      return False
    jump = math.hypot(blip.x - self._lock.x, blip.y - self._lock.y)
    if jump > self.track_radius_px:
      return False
    new_dist = self._dist_to_pivot(blip.x, blip.y, pivot)
    if (
      self._min_dist_seen is not None
      and self._min_dist_seen < self.close_hold_px
      and new_dist > cur_dist + self.max_regress_px
    ):
      return False
    if cur_dist < self.close_hold_px and jump > self.close_track_radius_px:
      return False
    return True

  def track(
    self,
    blips: tuple[Blip, ...],
    pivot: tuple[float, float],
  ) -> TargetLock | None:
    if self._lock is None:
      return None

    cur_dist = self._update_min_dist(pivot)
    assert cur_dist is not None
    close = cur_dist < self.close_hold_px
    search_r = self.close_track_radius_px if close else self.track_radius_px

    tier_blips = [b for b in blips if b.tier == self._lock.tier]
    best: Blip | None = None
    best_jump = search_r
    for blip in tier_blips:
      jump = math.hypot(blip.x - self._lock.x, blip.y - self._lock.y)
      if jump >= best_jump:
        continue
      if not self._accept_blip(blip, pivot, cur_dist=cur_dist):
        continue
      best = blip
      best_jump = jump

    pinned = (
      self._min_dist_seen is not None
      and self._min_dist_seen <= self.pin_dist_px
    )
    if pinned:
      self._lock.pinned = True

    if best is not None:
      self._lock.x = best.x
      self._lock.y = best.y
      self._lock.lost_frames = 0
      self._lock.pinned = False
      return self._lock

    if pinned or self._lock.pinned:
      self._lock.lost_frames = 0
      return self._lock

    self._lock.lost_frames += 1
    limit = self.max_lost_frames_close if close else self.max_lost_frames
    if self._lock.lost_frames > limit:
      was_close = (
        self._min_dist_seen is not None
        and self._min_dist_seen < self.close_hold_px
      )
      self._lock = None
      if not was_close:
        self._min_dist_seen = None
      return None
    return self._lock

  def refresh(
    self,
    blips: tuple[Blip, ...],
    pivot: tuple[float, float],
    *,
    auto_lock: bool = False,
    track: bool = True,
  ) -> TargetLock | None:
    if self._lock is not None:
      return self.track(blips, pivot) if track else self._lock
    if auto_lock and blips and self._should_auto_lock():
      return self.lock_nearest(blips, pivot)
    return None

  def _should_auto_lock(self) -> bool:
    if self._min_dist_seen is not None and self._min_dist_seen < self.close_hold_px:
      return False
    return True

  def lock_next(self, blips: tuple[Blip, ...], pivot: tuple[float, float]) -> TargetLock | None:
    self._min_dist_seen = None
    self._lock = None
    return self.lock_nearest(blips, pivot)
