"""Regressão: lost-target sem limpar lock → linha verde congelada + SCAN_SPIN."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

_MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MB))

from v2.vendor.node_detector import MiningNode, NodeScanResult, TargetLock
from v2.brain.tick import Brain
from v2.core.types import ArrowState, FrameContext, HudState, Phase
from v2.navigation.pursuit_controller import PursuitController


def _cfg(**nav_extra):
  nav = {
    "control_mode": "camera",
    "arrive_px": 17,
    "min_pick_px": 16,
    "sticky_enter_px": 40,
    "lock_max_lost_frames": 3,
    "lock_reacquire_cooldown_s": 0.05,
    "camera": {
      "first_person": True,
      "align_only": False,
      "scan_spin_enabled": False,
    },
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


def _frame_ctx(*, scan, legacy) -> FrameContext:
  return FrameContext(
    tick=1,
    timestamp=0.0,
    minimap_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
    hud_bgr=np.zeros((40, 40, 3), dtype=np.uint8),
    pivot=(160.0, 156.0),
    arrow=ArrowState(
      pivot_x=160.0,
      pivot_y=156.0,
      tip_x=160.0,
      tip_y=140.0,
      facing_deg=-90.0,
      detected=legacy is not None,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.GOTO,
    meta={"scan": scan, "legacy_arrow": legacy, "first_person": True},
  )


def test_missing_arrow_keeps_goto_lock():
  """Frame sem arrow NÃO demote a SCAN com lock órfão."""
  brain = Brain(_cfg(), _detector())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit._lock = TargetLock(
    tier="gray",
    locked_x=220.0,
    locked_y=140.0,
    pick_distance_px=70.0,
    last_distance_px=70.0,
    node_id=9,
    locked_area=20.0,
    last_bearing_deg=20.0,
    lost_frames=0,
    min_seen_distance_px=70.0,
  )
  scan = NodeScanResult(
    nodes=[],
    target=None,
    masks={},
    player_x=160.0,
    player_y=156.0,
  )
  out = brain.tick(_frame_ctx(scan=scan, legacy=None), enabled=True, game_focus=True)
  assert brain.phase == Phase.GOTO
  assert brain.pursuit.v1_lock is not None
  assert out.action == "stop"
  assert "hold" in str(out.meta.get("nav_status", ""))


def test_ghost_lost_clears_lock_without_blacklist():
  pc = PursuitController(_cfg(lock_max_lost_frames=3), _detector())
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
  empty = NodeScanResult(
    nodes=[], target=None, masks={}, player_x=160.0, player_y=156.0
  )
  # lock_max_lost_frames floor is 3 → need lost > 3.
  for _ in range(3):
    out = pc._resolve_target_camera(empty)
    assert out is not None
    assert pc.v1_lock is not None
  out = pc._resolve_target_camera(empty)
  assert out is None
  assert pc.v1_lock is None
  # Sem soft-blacklist permanente — nó pode ser re-escolhido depois do TTL.
  assert pc._done_xy == []
  # Soft-avoid temporário do XY abandonado (anti thrash).
  assert pc._is_ghost_avoided(235.0, 134.5)


def test_lost_target_arms_cooldown_before_relock():
  brain = Brain(_cfg(lock_reacquire_cooldown_s=0.4), _detector())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  node = MiningNode(
    tier="gray",
    x=200,
    y=140,
    radius=4.0,
    area=28.0,
    distance_px=55,
    circularity=0.85,
  )
  scan = NodeScanResult(
    nodes=[node],
    target=None,
    masks={},
    player_x=160.0,
    player_y=156.0,
  )
  legacy = SimpleNamespace(
    pivot=lambda: (160.0, 156.0),
    arrow_tip_x=160.0,
    arrow_tip_y=140.0,
    arrow_angle_deg=-90.0,
  )
  ctx = _frame_ctx(scan=scan, legacy=legacy)
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  assert brain._lock_cooldown_until > time.perf_counter()
  assert brain._post_lost_spin_gate is True
  # Durante cooldown / spin-gate não trava de novo.
  out2 = brain.tick(ctx, enabled=True, game_focus=True)
  assert brain.phase == Phase.SCAN
  assert brain.pursuit.v1_lock is None
  assert out.action == "lost-target"
  assert out2.lock is None


def test_lost_target_spin_gate_allows_live_relock():
  """Pós lost: disco live travável re-trava após cooldown (sem exigir 360°)."""
  brain = Brain(
    _cfg(
      lock_reacquire_cooldown_s=0.05,
      ghost_avoid_ttl_s=5.0,
      camera={
        "first_person": True,
        "align_only": False,
        "scan_spin_enabled": True,
        "scan_spin_total_deg": 360.0,
        "scan_spin_pulse_deg": 15.0,
        "scan_spin_interval_ms": 1.0,
        "scan_spin_pause_ms": 0.0,
      },
    ),
    _detector(),
  )
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.walker.look_yaw_deg = MagicMock(return_value="spin-look--15")

  other = MiningNode(
    tier="gray",
    x=100.0,
    y=120.0,
    radius=4.0,
    area=28.0,
    distance_px=55,
    circularity=0.85,
  )
  scan = NodeScanResult(
    nodes=[other],
    target=None,
    masks={},
    player_x=160.0,
    player_y=156.0,
  )
  legacy = SimpleNamespace(
    pivot=lambda: (160.0, 156.0),
    arrow_tip_x=160.0,
    arrow_tip_y=140.0,
    arrow_angle_deg=-90.0,
  )
  brain.pursuit._push_ghost_avoid(235.0, 134.5)
  ctx = _frame_ctx(scan=scan, legacy=legacy)
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.action == "lost-target"
  assert brain._post_lost_spin_gate is True

  time.sleep(0.06)
  out_lock = brain.tick(ctx, enabled=True, game_focus=True)
  assert brain.phase == Phase.GOTO
  assert brain.pursuit.v1_lock is not None
  assert abs(float(brain.pursuit.v1_lock.locked_x) - 100.0) < 1.0
  assert brain._post_lost_spin_gate is False
  assert out_lock.action == "scan-goto" or brain.phase == Phase.GOTO


def test_lost_target_spin_gate_spins_when_no_candidates():
  """Sem nó travável, gate mantém spin (não re-trava XY evitável)."""
  brain = Brain(
    _cfg(
      lock_reacquire_cooldown_s=0.01,
      ghost_avoid_ttl_s=5.0,
      camera={
        "first_person": True,
        "align_only": False,
        "scan_spin_enabled": True,
        "scan_spin_total_deg": 45.0,
        "scan_spin_pulse_deg": 15.0,
        "scan_spin_interval_ms": 1.0,
        "scan_spin_pause_ms": 0.0,
      },
    ),
    _detector(),
  )
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.walker.look_yaw_deg = MagicMock(return_value="spin-look--15")

  only_avoided = MiningNode(
    tier="gray",
    x=235.0,
    y=134.5,
    radius=4.0,
    area=28.0,
    distance_px=75,
    circularity=0.85,
  )
  scan = NodeScanResult(
    nodes=[only_avoided],
    target=None,
    masks={},
    player_x=160.0,
    player_y=156.0,
  )
  legacy = SimpleNamespace(
    pivot=lambda: (160.0, 156.0),
    arrow_tip_x=160.0,
    arrow_tip_y=140.0,
    arrow_angle_deg=-90.0,
  )
  brain.pursuit._push_ghost_avoid(235.0, 134.5)
  ctx = _frame_ctx(scan=scan, legacy=legacy)
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.action == "lost-target"

  time.sleep(0.02)
  for _ in range(2):
    out_spin = brain.tick(ctx, enabled=True, game_focus=True)
    assert brain.phase == Phase.SCAN
    assert brain.pursuit.v1_lock is None
    assert "spin" in str(out_spin.action) or str(out_spin.action).startswith(
      "spin"
    )
