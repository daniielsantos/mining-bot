"""Ghost lock longe sem rematch → abandona e re-permite SCAN."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MB))

from node_detector import MiningNode, NodeScanResult, TargetLock
from v2.navigation.pursuit_controller import PursuitController


def _cfg(**nav_extra):
  nav = {
    "control_mode": "camera",
    "sticky_enter_px": 40,
    "sticky_live_reacquire_px": 22,
    "lock_max_lost_frames": 5,
    "lock_max_lost_frames_close": 45,
    "lock_min_circularity": 0.55,
    "lock_min_area": 8.0,
    "min_pick_px": 16,
    "camera": {"first_person": True, "align_only": False},
  }
  nav.update(nav_extra)
  return {"navigation": nav, "allowed_tiers": ["gray"]}


def _detector():
  det = MagicMock()
  det.allowed_tiers = ["gray"]

  def _from_lock(lock, player_x=0.0, player_y=0.0):
    return MiningNode(
      tier=lock.tier,
      x=lock.locked_x,
      y=lock.locked_y,
      radius=3.0,
      area=20.0,
      distance_px=float(
        ((lock.locked_x - player_x) ** 2 + (lock.locked_y - player_y) ** 2) ** 0.5
      ),
      circularity=0.8,
      ghost=True,
    )

  det.node_from_lock.side_effect = _from_lock
  det.make_target_lock.side_effect = lambda scan, node: TargetLock(
    tier=node.tier,
    locked_x=node.x,
    locked_y=node.y,
    pick_distance_px=node.distance_px,
    last_distance_px=node.distance_px,
    node_id=1,
    locked_area=node.area,
    last_bearing_deg=0.0,
    lost_frames=0,
    min_seen_distance_px=node.distance_px,
  )
  return det


def _scan(nodes, px=160.0, py=156.0):
  return NodeScanResult(
    nodes=list(nodes),
    target=None,
    masks={},
    player_x=px,
    player_y=py,
  )


def test_lock_nearest_rejects_sparkle_and_road():
  pc = PursuitController(_cfg(), _detector())
  sparkle = MiningNode(
    tier="gray", x=220, y=140, radius=1.4, area=4.0, distance_px=60, circularity=0.8
  )
  roadish = MiningNode(
    tier="gray", x=230, y=145, radius=7.0, area=50.0, distance_px=70, circularity=0.4
  )
  real = MiningNode(
    tier="gray", x=100, y=120, radius=4.0, area=28.0, distance_px=55, circularity=0.82
  )
  pick = pc.lock_nearest(_scan([sparkle, roadish, real]), facing_deg=-90.0)
  assert pick is not None
  assert pick.x == real.x
  assert pick.y == real.y


def test_far_ghost_abandon_after_lost_frames():
  pc = PursuitController(_cfg(lock_max_lost_frames=3), _detector())
  # Lock fantasma longe: sem blips live → _commit_lost até abandonar.
  pc._lock = TargetLock(
    tier="gray",
    locked_x=235.0,
    locked_y=134.5,
    pick_distance_px=75.0,
    last_distance_px=75.0,
    node_id=2,
    locked_area=20.0,
    last_bearing_deg=70.0,
    lost_frames=0,
    min_seen_distance_px=75.0,
  )
  pc._last_live_x = 235.0
  pc._last_live_y = 134.5
  empty = _scan([])
  for _ in range(3):
    out = pc._resolve_target_camera(empty)
    assert out is not None
    assert pc.v1_lock is not None
  out = pc._resolve_target_camera(empty)
  assert out is None
  assert pc.v1_lock is None


def test_far_last_live_sticky_rematch():
  """Flicker breve: disco reaparece perto do last-live → mantém lock."""
  pc = PursuitController(_cfg(sticky_live_reacquire_px=22), _detector())
  pc._lock = TargetLock(
    tier="gray",
    locked_x=235.0,
    locked_y=134.5,
    pick_distance_px=75.0,
    last_distance_px=75.0,
    node_id=2,
    locked_area=20.0,
    last_bearing_deg=70.0,
    lost_frames=0,
    min_seen_distance_px=75.0,
  )
  pc._last_live_x = 235.0
  pc._last_live_y = 134.5
  live = MiningNode(
    tier="gray",
    x=248.0,
    y=140.0,
    radius=4.0,
    area=22.0,
    distance_px=80.0,
    circularity=0.8,
  )
  out = pc._resolve_target_camera(_scan([live]))
  assert out is not None
  assert getattr(out, "ghost", False) is False
  assert abs(out.x - 248.0) < 0.1
  assert pc.v1_lock is not None
  assert pc.v1_lock.lost_frames == 0


def test_ghost_rescue_retargets_live_instead_of_abandon():
  """Lock fantasma sem rematch local, mas gray live na tela → retarget."""
  pc = PursuitController(_cfg(lock_max_lost_frames=3), _detector())
  pc._lock = TargetLock(
    tier="gray",
    locked_x=304.0,
    locked_y=191.0,
    pick_distance_px=145.0,
    last_distance_px=145.0,
    node_id=9,
    locked_area=20.0,
    last_bearing_deg=104.0,
    lost_frames=0,
    min_seen_distance_px=145.0,
  )
  pc._last_live_x = 304.0
  pc._last_live_y = 191.0
  near = MiningNode(
    tier="gray",
    x=140.0,
    y=170.0,
    radius=4.0,
    area=28.0,
    distance_px=28.0,
    circularity=0.85,
  )
  # lost_frames floor=3 → precisa lost > 3 para rescue.
  for _ in range(3):
    out = pc._resolve_target_camera(_scan([near]))
    assert out is not None
    assert pc.v1_lock is not None
  out = pc._resolve_target_camera(_scan([near]))
  assert out is not None
  assert getattr(out, "ghost", False) is False
  assert abs(out.x - 140.0) < 0.1
  assert pc.v1_lock is not None
  assert pc._is_ghost_avoided(304.0, 191.0)


def test_close_protect_does_not_abandon_ghost():
  pc = PursuitController(_cfg(lock_max_lost_frames=2), _detector())
  pc._lock = TargetLock(
    tier="gray",
    locked_x=170.0,
    locked_y=150.0,
    pick_distance_px=20.0,
    last_distance_px=12.0,
    node_id=3,
    locked_area=20.0,
    last_bearing_deg=10.0,
    lost_frames=0,
    min_seen_distance_px=12.0,
  )
  pc._final_approaching = True
  empty = _scan([])
  for _ in range(10):
    out = pc._resolve_target_camera(empty)
    assert out is not None
    assert pc.v1_lock is not None
