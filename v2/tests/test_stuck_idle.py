"""GOTO walk sem progresso de dist → D recover ×N → mark_stuck + SCAN."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

_MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MB))

from node_detector import MiningNode, NodeScanResult, TargetLock
from v2.brain.tick import Brain
from v2.core.types import ArrowState, FrameContext, HudState, Phase
from v2.navigation.pursuit_controller import PursuitController


def _cfg(**nav_extra):
  nav = {
    "control_mode": "camera",
    "arrive_px": 15,
    "stuck_idle_s": 1.5,
    "stuck_progress_px": 1.5,
    "stuck_align_deg": 12,
    "stuck_d_hold_ms": 2000,
    "stuck_d_max_attempts": 3,
    "stuck_avoid_ttl_s": 45.0,
    "stuck_avoid_radius_px": 40.0,
    "done_radius_px": 36.0,
    "lock_min_circularity": 0.5,
    "lock_min_area": 6.0,
    "min_pick_px": 16,
    "camera": {"first_person": True, "align_only": False, "walk_max_deg": 12},
  }
  nav.update(nav_extra)
  return {"navigation": nav, "allowed_tiers": ["gray"]}


def _detector():
  det = MagicMock()
  det.allowed_tiers = ["gray"]
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


def _node(x, y, *, dist=None, area=28.0, circ=0.8):
  px, py = 160.0, 156.0
  if dist is None:
    dist = math.hypot(x - px, y - py)
  return MiningNode(
    tier="gray",
    x=x,
    y=y,
    radius=4.0,
    area=area,
    distance_px=dist,
    circularity=circ,
  )


def _scan(nodes, px=160.0, py=156.0):
  return NodeScanResult(
    nodes=list(nodes),
    target=None,
    masks={},
    player_x=px,
    player_y=py,
  )


def test_stuck_min_dist_defaults_to_arrive():
  pc = PursuitController(_cfg(), MagicMock())
  assert pc.stuck_min_dist_px == pc.arrive_px == 15.0


def test_stuck_d_config_defaults():
  pc = PursuitController(_cfg(), MagicMock())
  assert pc.stuck_d_hold_ms == 2000.0
  assert pc.stuck_d_max_attempts == 3
  assert pc._stuck_d_attempts == 0


def test_stuck_idle_triggers_after_no_progress(monkeypatch):
  pc = PursuitController(_cfg(), MagicMock())
  t0 = 1000.0
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0
  )
  assert (
    pc.check_stuck_idle(
      60.0, bearing_deg=3.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  # Sem avanço significativo.
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 1.0
  )
  assert (
    pc.check_stuck_idle(
      59.5, bearing_deg=2.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 1.6
  )
  assert (
    pc.check_stuck_idle(
      59.2, bearing_deg=2.0, move_phase="idle", expecting_walk=True
    )
    is True
  )


def test_stuck_idle_fires_when_frozen_just_outside_arrive(monkeypatch):
  """Bug: dist=19 < arrive+10 used to disable stuck forever; must re-lock."""
  pc = PursuitController(_cfg(), MagicMock())
  assert pc.stuck_min_dist_px == 15.0
  t0 = 4000.0
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0
  )
  assert (
    pc.check_stuck_idle(
      19.0, bearing_deg=0.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 1.0
  )
  assert (
    pc.check_stuck_idle(
      19.0, bearing_deg=0.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 1.6
  )
  assert (
    pc.check_stuck_idle(
      19.0, bearing_deg=0.0, move_phase="idle", expecting_walk=True
    )
    is True
  )


def test_stuck_idle_progress_keeps_d_attempts(monkeypatch):
  """Queda de dist reinicia timer STUCK, mas preserva o budget D do lock."""
  pc = PursuitController(_cfg(), MagicMock())
  t0 = 2000.0
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0
  )
  pc.check_stuck_idle(
    70.0, bearing_deg=1.0, move_phase="idle", expecting_walk=True
  )
  pc._stuck_d_attempts = 2
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 1.0
  )
  # Queda >= stuck_progress_px → timer reinicia; D attempts ficam.
  assert (
    pc.check_stuck_idle(
      68.0, bearing_deg=1.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  assert pc._stuck_d_attempts == 2
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 2.4
  )
  # Ainda dentro do novo stuck_idle_s após o progress.
  assert (
    pc.check_stuck_idle(
      67.5, bearing_deg=1.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  assert pc._stuck_d_attempts == 2


def test_stuck_idle_skips_fine_align_and_arrived(monkeypatch):
  pc = PursuitController(_cfg(), MagicMock())
  t0 = 3000.0
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0
  )
  pc.check_stuck_idle(
    60.0, bearing_deg=1.0, move_phase="idle", expecting_walk=True
  )
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 5.0
  )
  assert (
    pc.check_stuck_idle(
      60.0, bearing_deg=1.0, move_phase="fine_align", expecting_walk=True
    )
    is False
  )
  # Já chegou (dist <= arrive / stuck_min) → stuck desligado.
  assert (
    pc.check_stuck_idle(
      15.0, bearing_deg=1.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  assert (
    pc.check_stuck_idle(
      14.0, bearing_deg=1.0, move_phase="idle", expecting_walk=True
    )
    is False
  )


def test_recover_stuck_d_arms_post_realign():
  """Após pulse D: flag realign + smooth heading limpo (sem clamp pré-strafe)."""
  pc = PursuitController(_cfg(), MagicMock())
  pc._smooth_heading = 2.0
  pc.walker = MagicMock()
  pc.walker.pulse_strafe_d = MagicMock(return_value="strafe-d-2000ms")

  action = pc.recover_stuck_d()

  assert action == "strafe-d-2000ms"
  assert pc._stuck_d_attempts == 1
  assert pc._need_post_stuck_realign is True
  assert pc._smooth_heading is None


def test_walk_after_stuck_d_forces_realign_before_w(monkeypatch):
  """
  Pós D com brg ainda 'pequeno': 1º tick = realign (sem W);
  só depois de liberar o flag o walk pode ligar W.
  """
  pc = PursuitController(_cfg(), MagicMock())
  calls: list[dict] = []

  def _tick(**kwargs):
    calls.append(dict(kwargs))
    if kwargs.get("force_realign"):
      return "realign"
    return "walk"

  pc.walker = MagicMock()
  pc.walker.tick = MagicMock(side_effect=_tick)
  pc._need_post_stuck_realign = True

  # Tick 1: força realign mesmo com |brg| < walk_max; depois libera flag.
  a1 = pc.walk(3.0, dist_px=40.0, target_dot=0.9)
  assert a1 == "realign"
  assert calls[0]["force_realign"] is True
  assert pc._need_post_stuck_realign is False

  # Tick 2: já alinhado → walk normal sem force.
  a2 = pc.walk(3.0, dist_px=40.0, target_dot=0.9)
  assert a2 == "walk"
  assert calls[1]["force_realign"] is False


def test_walk_after_stuck_d_keeps_realign_while_misaligned():
  """|brg| > stuck_align_deg → permanece em force_realign (sem liberar W)."""
  pc = PursuitController(_cfg(), MagicMock())
  pc.walker = MagicMock()
  pc.walker.tick = MagicMock(return_value="realign-+20")
  pc._need_post_stuck_realign = True

  action = pc.walk(35.0, dist_px=40.0, target_dot=0.5)
  assert action == "realign-+20"
  assert pc._need_post_stuck_realign is True
  pc.walker.tick.assert_called_once()
  assert pc.walker.tick.call_args.kwargs["force_realign"] is True


def test_brain_stuck_idle_d_recover_keeps_lock():
  """Primeiros stucks: D recover, permanece GOTO, sem mark_stuck."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_stuck = MagicMock()
  brain.pursuit._stuck_d_attempts = 0
  brain.pursuit.stuck_d_max_attempts = 3

  def _recover():
    brain.pursuit._stuck_d_attempts += 1
    brain.pursuit._need_post_stuck_realign = True
    return "strafe-d-2000ms"

  brain.pursuit.recover_stuck_d = MagicMock(side_effect=_recover)

  action = brain._handle_stuck_idle(dist_px=19.0)

  assert action == "strafe-d-2000ms"
  assert brain.phase == Phase.GOTO
  assert brain.pursuit._stuck_d_attempts == 1
  assert brain.pursuit._need_post_stuck_realign is True
  brain.pursuit.mark_stuck.assert_not_called()
  brain.pursuit.stop_walk.assert_called_once()
  brain.pursuit.recover_stuck_d.assert_called_once()


def test_brain_stuck_idle_three_d_then_switch():
  """3× D recover → 4º stuck blacklista e SCAN."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_stuck = MagicMock()
  brain.pursuit.stuck_d_max_attempts = 3
  brain.pursuit._stuck_d_attempts = 0

  def _recover():
    brain.pursuit._stuck_d_attempts += 1
    return "strafe-d-2000ms"

  brain.pursuit.recover_stuck_d = MagicMock(side_effect=_recover)

  for i in range(3):
    action = brain._handle_stuck_idle(dist_px=19.0)
    assert action == "strafe-d-2000ms"
    assert brain.phase == Phase.GOTO
    assert brain.pursuit._stuck_d_attempts == i + 1
    brain.pursuit.mark_stuck.assert_not_called()

  action = brain._handle_stuck_idle(dist_px=19.0)
  assert action == "stuck-idle-next"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  assert brain.pursuit._stuck_d_attempts == 0
  brain.pursuit.mark_stuck.assert_called_once()
  assert brain.pursuit.recover_stuck_d.call_count == 3


def test_three_d_then_switch_locks_other_node(monkeypatch):
  """
  Após 3× D no mesmo lock: mark_stuck blacklista XY, limpa lock,
  lock_nearest escolhe outro nó (não o stuck). Jitter de dist não zera budget.
  """
  pc = PursuitController(_cfg(), _detector())
  stuck = _node(200.0, 156.0, dist=40.0)
  other = _node(160.0, 100.0, dist=56.0)
  pc._lock = TargetLock(
    tier="gray",
    locked_x=stuck.x,
    locked_y=stuck.y,
    pick_distance_px=stuck.distance_px,
    last_distance_px=stuck.distance_px,
    node_id=7,
    locked_area=stuck.area,
    last_bearing_deg=0.0,
    lost_frames=0,
    min_seen_distance_px=stuck.distance_px,
  )
  pc._last_live_x = stuck.x
  pc._last_live_y = stuck.y
  pc._smooth_facing = -90.0
  pc.stuck_d_max_attempts = 3
  pc._stuck_d_attempts = 0

  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit = pc
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.walker = MagicMock()
  brain.pursuit.walker.e_held_for_mine = False
  brain.pursuit.walker.pulse_strafe_d = MagicMock(return_value="strafe-d-2000ms")

  for _ in range(3):
    assert brain._handle_stuck_idle(dist_px=40.0).startswith("strafe-d")
    assert brain.phase == Phase.GOTO
    assert pc._lock is not None

  assert pc._stuck_d_attempts == 3

  t0 = 5000.0
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0
  )
  pc._stuck_best_dist = 40.0
  pc._stuck_since = t0
  monkeypatch.setattr(
    "v2.navigation.pursuit_controller.time.perf_counter", lambda: t0 + 0.5
  )
  assert (
    pc.check_stuck_idle(
      38.0, bearing_deg=1.0, move_phase="idle", expecting_walk=True
    )
    is False
  )
  assert pc._stuck_d_attempts == 3

  action = brain._handle_stuck_idle(dist_px=40.0)
  assert action == "stuck-idle-next"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  assert pc._lock is None
  assert pc._is_stuck_avoided(stuck.x, stuck.y)

  pick = pc.lock_nearest(_scan([stuck, other]), facing_deg=-90.0)
  assert pick is not None
  assert pick.x == other.x
  assert pick.y == other.y
  assert pc._stuck_d_attempts == 0


def test_new_lock_resets_stuck_d_attempts():
  """lock_nearest zera o contador D do lock anterior."""
  pc = PursuitController(_cfg(), _detector())
  a = _node(200.0, 156.0, dist=40.0)
  b = _node(160.0, 100.0, dist=56.0)
  pc._stuck_d_attempts = 3
  pick = pc.lock_nearest(_scan([a, b]), facing_deg=-90.0)
  assert pick is not None
  assert pc._stuck_d_attempts == 0


def test_brain_stuck_idle_switches_lock():
  """Após esgotar D attempts, mark_stuck + SCAN como antes."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_stuck = MagicMock()
  brain.pursuit._stuck_d_attempts = 3
  brain.pursuit.stuck_d_max_attempts = 3

  action = brain._handle_stuck_idle(dist_px=19.0)

  assert action == "stuck-idle-next"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  assert brain.pursuit._stuck_d_attempts == 0
  brain.pursuit.mark_stuck.assert_called_once()
  brain.pursuit.stop_walk.assert_called_once()


def test_mark_stuck_lock_nearest_skips_same_xy():
  """Após STUCK no XY, lock_nearest não re-trava o mesmo nó (escolhe o outro)."""
  pc = PursuitController(_cfg(), _detector())
  stuck = _node(200.0, 156.0, dist=40.0)
  other = _node(160.0, 100.0, dist=56.0)
  pc._lock = TargetLock(
    tier="gray",
    locked_x=stuck.x,
    locked_y=stuck.y,
    pick_distance_px=stuck.distance_px,
    last_distance_px=stuck.distance_px,
    node_id=7,
    locked_area=stuck.area,
    last_bearing_deg=0.0,
    lost_frames=0,
    min_seen_distance_px=stuck.distance_px,
  )
  pc._last_live_x = stuck.x
  pc._last_live_y = stuck.y
  pc._smooth_facing = -90.0

  pc.mark_stuck(facing_deg=-90.0)

  assert pc._lock is None
  assert pc._is_stuck_avoided(stuck.x, stuck.y)
  assert any(
    math.hypot(dx - stuck.x, dy - stuck.y) < 1.0 for dx, dy in pc._done_xy
  )

  pick = pc.lock_nearest(_scan([stuck, other]), facing_deg=-90.0)
  assert pick is not None
  assert pick.x == other.x
  assert pick.y == other.y


def test_mark_stuck_only_node_returns_none_for_spin():
  """Só o nó stuck existe → lock_nearest=None (Brain cai em SCAN_SPIN)."""
  pc = PursuitController(_cfg(), _detector())
  stuck = _node(200.0, 156.0, dist=40.0)
  pc._lock = TargetLock(
    tier="gray",
    locked_x=stuck.x,
    locked_y=stuck.y,
    pick_distance_px=stuck.distance_px,
    last_distance_px=stuck.distance_px,
    node_id=7,
    locked_area=stuck.area,
    last_bearing_deg=0.0,
    lost_frames=0,
    min_seen_distance_px=stuck.distance_px,
  )
  pc._last_live_x = stuck.x
  pc._last_live_y = stuck.y
  pc.mark_stuck(facing_deg=-90.0)

  assert pc.lock_nearest(_scan([stuck]), facing_deg=-90.0) is None


def test_mark_stuck_polar_survives_minimap_rotation():
  """
  SCAN_SPIN gira o minimapa heading-up: XY absoluto do nó muda, mas
  dist + bearing rel. facing (mundo) permanecem → ainda skip.
  """
  pc = PursuitController(_cfg(), _detector())
  facing0 = -90.0
  px, py = 160.0, 156.0
  stuck_x, stuck_y = 160.0, 116.0  # 40px "north"
  pc._lock = TargetLock(
    tier="gray",
    locked_x=stuck_x,
    locked_y=stuck_y,
    pick_distance_px=40.0,
    last_distance_px=40.0,
    node_id=3,
    locked_area=28.0,
    last_bearing_deg=math.degrees(math.atan2(stuck_y - py, stuck_x - px)),
    lost_frames=0,
    min_seen_distance_px=40.0,
  )
  pc._last_live_x = stuck_x
  pc._last_live_y = stuck_y
  pc.mark_stuck(facing_deg=facing0)

  # Player/câmera +45°: minimapa gira e facing atualiza juntos.
  turn = 45.0
  ang = math.radians(turn)
  dx0, dy0 = stuck_x - px, stuck_y - py
  c, s = math.cos(ang), math.sin(ang)
  rot_x = px + dx0 * c - dy0 * s
  rot_y = py + dx0 * s + dy0 * c
  facing1 = facing0 + turn
  rotated = _node(rot_x, rot_y, dist=40.0)
  other = _node(100.0, 156.0, dist=60.0)

  assert math.hypot(rot_x - stuck_x, rot_y - stuck_y) > 28.0

  pick = pc.lock_nearest(
    _scan([rotated, other], px=px, py=py), facing_deg=facing1
  )
  assert pick is not None
  assert pick.x == other.x
  assert abs(pick.y - other.y) < 0.1


def _lock_on(pc: PursuitController, node: MiningNode, *, bearing=0.0, facing=-90.0):
  pc._lock = TargetLock(
    tier="gray",
    locked_x=node.x,
    locked_y=node.y,
    pick_distance_px=node.distance_px,
    last_distance_px=node.distance_px,
    node_id=1,
    locked_area=node.area,
    last_bearing_deg=bearing,
    lost_frames=0,
    min_seen_distance_px=node.distance_px,
  )
  pc._last_live_x = node.x
  pc._last_live_y = node.y
  pc._smooth_facing = facing


def test_two_stucks_both_avoided():
  """Stuck A → lock B → stuck B → A e B evitados; escolhe C."""
  pc = PursuitController(_cfg(), _detector())
  a = _node(200.0, 156.0, dist=40.0)
  b = _node(160.0, 100.0, dist=56.0)
  c = _node(100.0, 156.0, dist=60.0)

  _lock_on(pc, a, bearing=0.0)
  pc.mark_stuck(facing_deg=-90.0)
  assert pc._is_stuck_avoided(a.x, a.y)
  assert len(pc._stuck_done_xy) == 1
  assert len(pc._stuck_avoid_xy) == 1

  _lock_on(pc, b, bearing=-90.0)
  pc.mark_stuck(facing_deg=-90.0)
  assert pc._is_stuck_avoided(a.x, a.y)
  assert pc._is_stuck_avoided(b.x, b.y)
  assert len(pc._stuck_done_xy) == 2
  assert len(pc._stuck_avoid_xy) == 2

  pick = pc.lock_nearest(_scan([a, b, c]), facing_deg=-90.0)
  assert pick is not None
  assert pick.x == c.x
  assert pick.y == c.y


def test_clear_stuck_blacklist_after_mine_success():
  """Após mine success, A/B stuck voltam a ser lockáveis; nó minado permanece done."""
  pc = PursuitController(_cfg(), _detector())
  a = _node(200.0, 156.0, dist=40.0)
  b = _node(160.0, 100.0, dist=56.0)
  mined = _node(120.0, 120.0, dist=50.0)

  _lock_on(pc, a)
  pc.mark_stuck(facing_deg=-90.0)
  _lock_on(pc, b)
  pc.mark_stuck(facing_deg=-90.0)
  assert len(pc._stuck_done_xy) == 2

  # Simula mine success noutro nó (como _enter_ready: clear depois mark_done).
  _lock_on(pc, mined)
  pc.clear_stuck_blacklist()
  pc.mark_done()

  assert pc._stuck_done_xy == []
  assert pc._stuck_avoid_xy == []
  assert pc._stuck_polar_avoid == []
  assert not pc._is_stuck_avoided(a.x, a.y)
  assert not pc._is_stuck_avoided(b.x, b.y)
  # Stuck XY saíram de _done_xy; o minado ficou.
  assert not any(
    math.hypot(dx - a.x, dy - a.y) < 1.0 for dx, dy in pc._done_xy
  )
  assert not any(
    math.hypot(dx - b.x, dy - b.y) < 1.0 for dx, dy in pc._done_xy
  )
  assert any(
    math.hypot(dx - mined.x, dy - mined.y) < 1.0 for dx, dy in pc._done_xy
  )

  pick = pc.lock_nearest(_scan([a, b]), facing_deg=-90.0)
  assert pick is not None
  assert (pick.x, pick.y) in {(a.x, a.y), (b.x, b.y)}


def test_enter_ready_clears_stuck_blacklist():
  """READY_INTERACT (Mining ore) chama clear_stuck_blacklist antes de mark_done."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.e_held_for_mine = False
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  order: list[str] = []

  def _clear():
    order.append("clear")

  def _done():
    order.append("done")

  brain.pursuit.clear_stuck_blacklist = MagicMock(side_effect=_clear)
  brain.pursuit.mark_done = MagicMock(side_effect=_done)

  action = brain._enter_ready(reason="test", score=0.95)

  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  assert brain.walker.e_held_for_mine is True
  brain.walker.begin_probe_e.assert_called()
  brain.walker.keep_e_for_mine.assert_called()
  assert order == ["clear", "done"]


def test_handle_stuck_idle_skips_when_e_held():
  """E held (mine) → sem D strafe / mark_stuck."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain.walker.e_held_for_mine = True
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.recover_stuck_d = MagicMock(return_value="strafe-d-2000ms")
  brain.pursuit._reset_stuck_idle = MagicMock()

  action = brain._handle_stuck_idle(dist_px=40.0)
  assert action == "stuck-skip-mining"
  brain.pursuit.recover_stuck_d.assert_not_called()
  brain.pursuit.stop_walk.assert_called()


def test_ready_mine_hold_tick_never_calls_stuck(monkeypatch):
  """READY mine-hold: check_stuck_idle / recover_stuck_d não rodam."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.READY_INTERACT
  brain._mining_label_seen = True
  brain._mining_had_strong = True
  brain._ore_peak = 0.97
  brain.walker.e_held_for_mine = True
  brain.walker.keep_e_for_mine = MagicMock()
  brain.walker.begin_probe_e = MagicMock()
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.check_stuck_idle = MagicMock(return_value=True)
  brain.pursuit.recover_stuck_d = MagicMock(return_value="strafe-d-2000ms")
  brain._handle_stuck_idle = MagicMock(return_value="stuck-idle-d-recover")

  hit = SimpleNamespace(found=True, raw_hit=True, score=0.95, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  ctx = FrameContext(
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
      detected=True,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.READY_INTERACT,
    meta={"scan": None, "legacy_arrow": object(), "first_person": True},
  )
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.phase == Phase.READY_INTERACT
  assert out.action == "mine-hold"
  brain.pursuit.check_stuck_idle.assert_not_called()
  brain._handle_stuck_idle.assert_not_called()
  brain.pursuit.recover_stuck_d.assert_not_called()


def test_goto_ore_strong_alone_does_not_ready(monkeypatch):
  """GOTO com ore≥match mas sem e_held: NÃO READY (sem close-walk/probe)."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"] = {
    "match_threshold": 0.70,
    "hold_min": 0.70,
  }
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.walker.e_held_for_mine = False
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit._lock = None
  brain.pursuit.check_stuck_idle = MagicMock(return_value=False)
  brain.pursuit.recover_stuck_d = MagicMock(return_value="strafe-d-2000ms")
  brain.pursuit.clear_stuck_blacklist = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.pursuit.walk = MagicMock(return_value="fine-aligned")
  brain.pursuit.evaluate = MagicMock(
    return_value=SimpleNamespace(
      move_phase="fine_align",
      bearing_deg=2.0,
      dist_px=16.0,
      target_dot=0.9,
      target=object(),
      display_lock=None,
      arrived=False,
      aligned=False,
      nav_status="",
    )
  )

  hit = SimpleNamespace(found=False, raw_hit=False, score=0.70, near_miss=True)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  node = _node(176.0, 156.0, dist=16.0)
  legacy = SimpleNamespace(facing_deg=-90.0)
  ctx = FrameContext(
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
      detected=True,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.GOTO,
    meta={"scan": _scan([node]), "legacy_arrow": legacy, "first_person": True},
  )
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.phase == Phase.GOTO
  assert out.action != "ready-interact"
  assert brain.walker.e_held_for_mine is False
  brain.pursuit.mark_done.assert_not_called()
  brain.pursuit.walk.assert_called()


def test_goto_fine_align_ore_070_stays_fine_align(monkeypatch):
  """PAROU/fine-align + ore=0.70 ambient → continua fine-align, nunca READY."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"] = {
    "match_threshold": 0.70,
    "hold_min": 0.70,
  }
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain._announced_fine_align = False
  brain.walker.e_held_for_mine = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit._lock = None
  brain.pursuit.walk = MagicMock(return_value="fine-align")
  brain.pursuit.evaluate = MagicMock(
    return_value=SimpleNamespace(
      move_phase="fine_align",
      bearing_deg=1.5,
      dist_px=16.0,
      target_dot=0.95,
      target=object(),
      display_lock=None,
      arrived=False,
      aligned=False,
      nav_status="",
    )
  )
  hit = SimpleNamespace(found=False, raw_hit=False, score=0.70, near_miss=True)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  node = _node(176.0, 156.0, dist=16.0)
  legacy = SimpleNamespace(facing_deg=-90.0)
  ctx = FrameContext(
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
      detected=True,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.GOTO,
    meta={"scan": _scan([node]), "legacy_arrow": legacy, "first_person": True},
  )
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.phase == Phase.GOTO
  assert brain._announced_fine_align is True
  assert out.action == "fine-align"
  assert "STILL_MINING" not in str(out.meta.get("nav_status") or "")


def test_goto_still_mining_only_with_e_held_and_label(monkeypatch):
  """STILL_MINING resume: e_held + label present → READY; ore alone → não."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"] = {
    "match_threshold": 0.70,
    "hold_min": 0.70,
  }
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.walker.e_held_for_mine = True
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock()
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit._lock = None
  brain.pursuit.clear_stuck_blacklist = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.pursuit.evaluate = MagicMock(return_value=None)

  hit = SimpleNamespace(found=False, raw_hit=False, score=0.85, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)
  assert brain._can_resume_still_mining(hit) is True

  ctx = FrameContext(
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
      detected=True,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.GOTO,
    meta={"scan": None, "legacy_arrow": object(), "first_person": True},
  )
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.phase == Phase.READY_INTERACT
  assert out.action == "ready-interact"


def test_ready_only_after_probe_with_present(monkeypatch):
  """Após close-walk: READY só com label present durante/após probe E."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"] = {
    "match_threshold": 0.70,
    "hold_min": 0.70,
  }
  cfg["navigation"]["final_probe_e_ms"] = 750
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.walker.e_held_for_mine = False
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit._lock = None
  brain.pursuit.begin_final_approach = MagicMock()
  brain.pursuit.clear_stuck_blacklist = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.pursuit.evaluate = MagicMock(
    return_value=SimpleNamespace(
      move_phase="close_done",
      bearing_deg=0.0,
      dist_px=8.0,
      target_dot=1.0,
      target=object(),
      display_lock=None,
      arrived=True,
      aligned=True,
      nav_status="",
    )
  )

  # Ambient during entry → still probe, not READY from ore alone.
  weak = SimpleNamespace(found=False, raw_hit=False, score=0.70, near_miss=True)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: weak)

  node = _node(168.0, 156.0, dist=8.0)
  legacy = SimpleNamespace(facing_deg=-90.0)
  ctx = FrameContext(
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
      detected=True,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.GOTO,
    meta={"scan": _scan([node]), "legacy_arrow": legacy, "first_person": True},
  )
  out1 = brain.tick(ctx, enabled=True, game_focus=True)
  assert out1.phase == Phase.FINAL_APPROACH
  assert brain.walker.e_held_for_mine is False or brain._fa_sub == "holding_e"

  # Mid-probe with present → READY.
  present = SimpleNamespace(
    found=True, raw_hit=True, score=0.92, near_miss=False
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: present)
  brain._fa_sub = "holding_e"
  brain._final_probe_until = 1e18
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")

  out2 = brain.tick(ctx, enabled=True, game_focus=True)
  assert out2.phase == Phase.READY_INTERACT
  assert brain.walker.e_held_for_mine is True


def test_goto_ore_strong_resumes_ready_no_stuck(monkeypatch):
  """Legado: ore forte sozinho NÃO resume READY (precisa e_held + label)."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.GOTO
  brain._allow_auto_lock = False
  brain.walker.e_held_for_mine = False
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit._lock = None
  brain.pursuit.check_stuck_idle = MagicMock(return_value=False)
  brain.pursuit.recover_stuck_d = MagicMock(return_value="strafe-d-2000ms")
  brain.pursuit.clear_stuck_blacklist = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.pursuit.evaluate = MagicMock(return_value=None)

  hit = SimpleNamespace(found=False, raw_hit=False, score=0.92, near_miss=True)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  ctx = FrameContext(
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
      detected=True,
    ),
    blips=(),
    hud=HudState(mining_active=False, progress_pct=None, has_label=False),
    phase=Phase.GOTO,
    meta={"scan": None, "legacy_arrow": object(), "first_person": True},
  )
  out = brain.tick(ctx, enabled=True, game_focus=True)
  assert out.phase == Phase.GOTO
  assert brain.walker.e_held_for_mine is False
  brain.pursuit.mark_done.assert_not_called()
  brain.pursuit.recover_stuck_d.assert_not_called()
