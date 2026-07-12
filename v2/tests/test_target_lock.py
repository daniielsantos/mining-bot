"""Testes v2 — target lock."""

from v2.core.types import Blip
from v2.navigation.target_lock import TargetLocker


def test_lock_nearest_skips_too_close():
  locker = TargetLocker(min_pick_px=20.0)
  blips = (
    Blip(x=100, y=100, tier="gray", radius=4, distance_px=5),
    Blip(x=150, y=80, tier="gray", radius=4, distance_px=50),
  )
  lock = locker.lock_nearest(blips, pivot=(100, 100))
  assert lock is not None
  assert lock.x == 150


def test_track_rejects_regress_when_close():
  locker = TargetLocker(close_hold_px=40, min_pick_px=0)
  locker._lock = locker.lock_nearest(
    (Blip(x=170, y=140, tier="gray", radius=4, distance_px=30),),
    pivot=(163, 156),
  )
  assert locker.lock is not None
  locker._min_dist_seen = 16.0
  pivot = (163, 156)
  far = Blip(x=285, y=164, tier="gray", radius=4, distance_px=120)
  near = Blip(x=164, y=141, tier="gray", radius=4, distance_px=15)
  locker.track((far, near), pivot)
  assert locker.lock.x == 164
  assert locker.lock.y == 141
