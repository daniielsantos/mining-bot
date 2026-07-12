"""Testes do abort/restart no FINAL_APPROACH (pulse_max → SCAN + auto-lock)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# Prefer mining_bot/keyboard_input over repo-root stub.
_MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_MB))

import numpy as np

from v2.brain.tick import Brain
from v2.core.types import Phase


def _cfg():
  return {
    "navigation": {
      "final_pulse_max": 5,
      "final_approach_timeout_s": 0,
      "final_pulse_w_ms": 350,
      "final_wait_after_w_ms": 300,
      "final_wait_after_e_ms": 0,
      "final_wait_before_e_ms": 0,
      "final_probe_e_ms": 50,
      "mining_ore": {
        "match_threshold": 0.99,
        "confirm_frames": 99,
        "template": "assets/mining_ore_label.png",
      },
    }
  }


def _ctx():
  return SimpleNamespace(
    hud_bgr=np.zeros((90, 520, 3), dtype=np.uint8),
    meta={},
  )


def _ready_brain(cfg=None, *, confirm=3):
  cfg = cfg or _cfg()
  ore = cfg["navigation"]["mining_ore"]
  ore.setdefault("match_threshold", 0.85)
  ore.setdefault("hold_min", 0.75)
  ore.setdefault("gone_threshold", 0.60)
  ore.setdefault("gone_drop_from_peak", 0.20)
  ore.setdefault("score_smooth_frames", 5)
  ore.setdefault("mine_hold_timeout_s", 0)  # off in unit tests unless set
  ore.setdefault("mine_hold_below_match_s", 0)
  ore["gone_confirm_frames"] = confirm
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.READY_INTERACT
  brain._mining_label_seen = True
  brain._mining_had_strong = True
  brain._ore_gone_streak = 0
  brain._ore_peak = 0.0
  brain._mining_ready_at = 0.0  # started; timeouts off via cfg
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.e_held_for_mine = True
  brain.walker.clear_e_hold = MagicMock()
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  return brain


def test_pulse_max_restarts_not_ready(monkeypatch):
  """Após final_pulse_max ciclos W+E: miss → mark_done + SCAN (não READY)."""
  import time as time_mod

  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = brain.final_pulse_max  # já esgotou W+E
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = time_mod.perf_counter() - 1.0
  brain._allow_auto_lock = False
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.clear_e_hold = MagicMock()
  miss = SimpleNamespace(found=False, raw_hit=False, score=0.1, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )

  assert action == "probe-miss-restart"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  brain.pursuit.mark_done.assert_called_once()
  assert "SCAN" in nav


def test_pulse_max_close_retries_same_target(monkeypatch):
  """Pulse-max miss perto do lock → mark_done + SCAN (outro alvo)."""
  import time as time_mod

  cfg = _cfg()
  cfg["navigation"]["final_abort_retry_px"] = 40
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = brain.final_pulse_max
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = time_mod.perf_counter() - 1.0
  brain._allow_auto_lock = False
  lock = SimpleNamespace(last_distance_px=22.0)
  brain.pursuit._lock = lock
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.clear_e_hold = MagicMock()
  miss = SimpleNamespace(found=False, raw_hit=False, score=0.1, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace(dist_px=22.0)
  )

  assert action == "probe-miss-restart"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  brain.pursuit.mark_done.assert_called_once()
  brain.pursuit.reset.assert_not_called()


def test_pulse_max_far_still_scans(monkeypatch):
  """Pulse-max miss longe → mark_done + SCAN + auto-lock."""
  import time as time_mod

  cfg = _cfg()
  cfg["navigation"]["final_abort_retry_px"] = 40
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = brain.final_pulse_max
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = time_mod.perf_counter() - 1.0
  brain._allow_auto_lock = False
  brain.pursuit._lock = SimpleNamespace(last_distance_px=118.0)
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.reset = MagicMock()
  brain.pursuit.mark_done = MagicMock()
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.clear_e_hold = MagicMock()
  miss = SimpleNamespace(found=False, raw_hit=False, score=0.1, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace(dist_px=118.0)
  )

  assert action == "probe-miss-restart"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  brain.pursuit.mark_done.assert_called_once()


def test_final_approach_starts_with_probe_e(monkeypatch):
  """Entrada em FINAL_APPROACH: primeiro probe-E, sem pulse-W."""
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = None
  brain._pulse_count = 0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-150ms")

  miss = SimpleNamespace(found=False, score=0.1, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )

  assert action == "probe-e-down"
  assert brain._fa_sub == "holding_e"
  assert brain._pulse_count == 0
  brain.walker.begin_probe_e.assert_called_once()
  brain.walker.pulse_forward.assert_not_called()
  assert "holding_e" in nav


def test_standstill_wait_before_initial_and_loop_e(monkeypatch):
  """Antes de cada E: parado + final_wait_before_e_ms, depois keydown."""
  import time as time_mod

  cfg = _cfg()
  cfg["navigation"]["final_wait_before_e_ms"] = 200
  cfg["navigation"]["final_wait_after_w_ms"] = 10
  cfg["navigation"]["final_probe_e_ms"] = 50
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = None
  brain._pulse_count = 0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-350ms")

  miss = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.1
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  # 1º tick pós close-walk: standstill, sem E ainda.
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-wait"
  assert brain._fa_sub == "e"
  brain.walker.begin_probe_e.assert_not_called()
  brain.pursuit.stop_walk.assert_called()
  assert "wait-before-e" in nav

  # Após 200ms: keydown E.
  brain._fa_sub_at = time_mod.perf_counter() - 0.25
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-down"
  assert brain._fa_sub == "holding_e"
  brain.walker.begin_probe_e.assert_called_once()

  # Miss → W → wait_w → standstill de novo antes do 2º E.
  brain._fa_sub_at = time_mod.perf_counter() - 0.1
  action, _ = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-miss"
  assert brain._fa_sub == "w"

  action, _ = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "pulse-w-350ms"
  assert brain._fa_sub == "wait_w"

  brain._fa_sub_at = time_mod.perf_counter() - 0.05
  brain.walker.begin_probe_e.reset_mock()
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-wait"
  assert brain._fa_sub == "e"
  brain.walker.begin_probe_e.assert_not_called()
  assert "wait-before-e" in nav

  brain._fa_sub_at = time_mod.perf_counter() - 0.25
  action, _ = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-down"
  assert brain._fa_sub == "holding_e"
  brain.walker.begin_probe_e.assert_called_once()


def test_final_w_wait_e_cycle_after_initial_miss(monkeypatch):
  """Miss inicial → W → wait → E; present mid-hold → READY."""
  import time as time_mod

  cfg = _cfg()
  cfg["navigation"]["final_probe_e_ms"] = 50
  cfg["navigation"]["final_wait_after_w_ms"] = 10
  cfg["navigation"]["final_pulse_w_ms"] = 350
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = time_mod.perf_counter()
  brain._pulse_count = 0
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = time_mod.perf_counter() - 0.1
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-350ms")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  miss = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.1
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  action, _ = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-miss"
  assert brain._fa_sub == "w"

  action, _ = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "pulse-w-350ms"
  assert brain._fa_sub == "wait_w"
  assert brain._pulse_count == 1

  brain._fa_sub_at = time_mod.perf_counter() - 0.05
  action, _ = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-down"
  assert brain._fa_sub == "holding_e"

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.80
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  assert "READY" in nav
  brain.walker.pulse_forward.assert_called_once()


def test_final_soft_engage_does_not_skip_initial_probe_e(monkeypatch):
  """
  Abaixo de hold_min no 1º tick: ainda faz probe E (não READY).
  Presente (≥hold_min) commitaria — aqui 0.40 < hold.
  """
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  cfg["navigation"]["mining_ore"]["final_keep_min"] = 0.55
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = None
  brain._pulse_count = 0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-380ms")
  brain.walker.e_held_for_mine = False

  noise = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.40
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: noise)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )

  assert action == "probe-e-down"
  assert brain._fa_sub == "holding_e"
  assert brain.phase == Phase.FINAL_APPROACH
  brain.pursuit.mark_done.assert_not_called()
  brain.walker.begin_probe_e.assert_called_once()
  brain.walker.pulse_forward.assert_not_called()


def test_ore_engage_false_outside_final():
  """engage_threshold nunca true em GOTO — não pula fine/close."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.GOTO
  hit = SimpleNamespace(found=False, raw_hit=False, score=0.70)
  assert brain._ore_engage(hit) is False
  assert brain._ore_final_keep(hit) is False
  brain.phase = Phase.FINAL_APPROACH
  brain._fa_sub = "e"  # antes do probe: present não bloqueia W / READY
  assert brain._ore_engage(hit) is True  # log quase / present
  assert brain._ore_final_keep(hit) is False
  brain._fa_sub = "holding_e"
  assert brain._ore_final_keep(hit) is True  # present após probe
  strong = SimpleNamespace(found=False, raw_hit=False, score=0.90)
  assert brain._ore_final_keep(strong) is True


def test_initial_probe_e_pulse_wait_before_first_w(monkeypatch):
  """Após probe-E inicial miss: vai para W (não COOLDOWN ainda)."""
  import time as time_mod

  cfg = _cfg()
  cfg["navigation"]["final_probe_e_ms"] = 50
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = time_mod.perf_counter()
  brain._pulse_count = 0
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = time_mod.perf_counter() - 0.1  # hold já esgotou
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.clear_e_hold = MagicMock()
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-350ms")

  miss = SimpleNamespace(found=False, score=0.1, near_miss=False)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: miss)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-miss"
  assert brain.phase == Phase.FINAL_APPROACH
  assert brain._fa_sub == "w"
  brain.walker.end_probe_e.assert_called()
  brain.walker.pulse_forward.assert_not_called()  # W no próximo tick


def test_ready_only_on_mining_ore(monkeypatch):
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 3
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = 0.0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.pursuit.reset = MagicMock()
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  hit = SimpleNamespace(found=True, score=0.95, raw_hit=True)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  brain.pursuit.mark_done.assert_called_once()
  assert "READY" in nav
  assert brain._mining_label_seen is True
  assert brain._mining_had_strong is True
  assert brain._ore_peak >= 0.95


def test_final_score_above_thr_enters_ready_no_pulse_w(monkeypatch):
  """
  Bug: ore≥thr logado como 'quase' (contrast/confirm falhou) e FINAL
  continuava pulse-W. score≥match deve ir a READY sem W.
  """
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["confirm_frames"] = 99
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 1
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = 0.0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-150ms")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  # found=False / raw_hit=False — exatamente o caso 'quase' com score alto.
  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=True, score=0.90
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  brain.walker.pulse_forward.assert_not_called()
  # holding_e + flag False: begin+keep ANTES de stop (não só keep).
  brain.walker.begin_probe_e.assert_called()
  brain.walker.keep_e_for_mine.assert_called()
  assert brain.walker.e_held_for_mine is True
  brain.pursuit.stop_walk.assert_called()
  assert "READY" in nav


def test_final_wait_e_score_above_thr_no_pulse_w(monkeypatch):
  """Present em wait_e: begin E + READY (nunca pulse-W)."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 1
  brain._fa_sub = "wait_e"
  brain._fa_sub_at = 0.0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-350ms")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False
  brain.walker.can_pulse = MagicMock(return_value=True)

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.96
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  brain.walker.pulse_forward.assert_not_called()
  brain.walker.begin_probe_e.assert_called()
  brain.walker.keep_e_for_mine.assert_called()
  assert brain.walker.e_held_for_mine is True
  assert "READY" in nav


def test_final_mid_score_holding_e_commits_ready(monkeypatch):
  """
  Present unificado: ore~0.61 (≥hold_min) durante holding_e → READY,
  para W, keep E — mesmo sinal do mine-hold (não exige match 0.85).
  """
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 2
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = 0.0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-380ms")

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.61
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  assert brain._ore_label_present(hit) is True
  assert brain._ore_strong(hit) is False
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  assert "READY" in nav
  brain.walker.pulse_forward.assert_not_called()
  assert brain.walker.e_held_for_mine is True


def test_final_below_hold_holding_e_continues_to_w(monkeypatch):
  """Abaixo de hold_min sem found: hold até probe_ms → W (ainda há pulsos)."""
  import time as time_mod

  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  cfg["navigation"]["final_probe_e_ms"] = 50
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 2
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = time_mod.perf_counter()
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-380ms")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.end_probe_e = MagicMock(return_value="probe-e-up")
  brain.walker.clear_e_hold = MagicMock()
  brain.walker.e_held_for_mine = False

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.40
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-hold"
  assert brain.phase == Phase.FINAL_APPROACH

  brain._fa_sub_at = time_mod.perf_counter() - 0.1
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "probe-e-miss"
  assert brain.phase == Phase.FINAL_APPROACH
  assert brain._fa_sub == "w"
  brain.walker.end_probe_e.assert_called()
  brain.walker.pulse_forward.assert_not_called()
  brain.pursuit.mark_done.assert_not_called()


def test_final_present_wait_e_commits_ready(monkeypatch):
  """Após probe-up, score≥hold_min em wait_e → READY (não pulse-W)."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["final_wait_after_e_ms"] = 0
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 3
  brain._fa_sub = "wait_e"
  brain._fa_sub_at = 0.0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-380ms")

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.65
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  brain.walker.pulse_forward.assert_not_called()


def test_final_present_on_w_blocks_pulse(monkeypatch):
  """Present em sub-step w: READY / stop W — não pulse."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 1
  brain._fa_sub = "w"
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.can_pulse = MagicMock(return_value=True)
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-380ms")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.58
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  brain.walker.pulse_forward.assert_not_called()


def test_ready_present_wait_w_commits(monkeypatch):
  """Present 0.55–0.73 em wait_w (pós probe) → READY."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["final_wait_after_w_ms"] = 10_000
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 2
  brain._fa_sub = "wait_w"
  brain._fa_sub_at = time.perf_counter()
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  for score in (0.55, 0.63, 0.73):
    brain.phase = Phase.FINAL_APPROACH
    brain._final_started_at = 0.0
    brain._fa_sub = "wait_w"
    brain.walker.e_held_for_mine = False
    hit = SimpleNamespace(
      found=False, raw_hit=False, near_miss=False, score=score
    )
    monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx, h=hit: h)
    action, nav = brain._tick_final_approach(
      _ctx(), active=True, pursuit=SimpleNamespace()
    )
    assert brain.phase == Phase.READY_INTERACT, f"score={score}"
    assert action == "ready-interact"


def test_ready_strong_ore_holding_e_immediate(monkeypatch):
  """Strong ore (≥match) durante holding_e → READY no 1º frame."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  cfg["navigation"]["mining_ore"]["ready_confirm_frames"] = 5
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 1
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = 0.0
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.90
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  assert "READY" in nav


def test_final_present_keep_blocks_w(monkeypatch):
  """hold_min present (≥0.55) sem found bloqueia W após probe — READY."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.55
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  cfg["navigation"]["mining_ore"]["final_keep_min"] = 0.55
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 0
  brain._fa_sub = "w"
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.pursuit.mark_done = MagicMock()
  brain.walker.can_pulse = MagicMock(return_value=True)
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-150ms")
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.e_held_for_mine = False

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.62
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  assert brain._ore_final_keep(hit) is True
  assert brain._ore_label_present(hit) is True
  assert brain._ore_strong(hit) is False
  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  brain.walker.pulse_forward.assert_not_called()


def test_final_below_engage_still_allows_pulse_w(monkeypatch):
  """Sub-step w sem label: pulse W e entra wait_w."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["engage_threshold"] = 0.55
  cfg["navigation"]["mining_ore"]["final_keep_min"] = 0.55
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 0
  brain._fa_sub = "w"
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.clear_e_hold = MagicMock()
  brain.walker.pulse_forward = MagicMock(return_value="pulse-w-350ms")

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=False, score=0.40
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "pulse-w-350ms"
  assert brain.phase == Phase.FINAL_APPROACH
  assert brain._fa_sub == "wait_w"
  assert brain._pulse_count == 1
  brain.walker.pulse_forward.assert_called_once()
  brain.walker.pulse_forward.assert_called_with(hold_ms=350.0)


def test_ready_mine_hold_stops_walk(monkeypatch):
  """READY mine-hold força stop_walk — sem W/stuck durante mining."""
  brain = _ready_brain(confirm=3)
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  hit = SimpleNamespace(found=True, raw_hit=True, score=0.97)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mine-hold"
  brain.pursuit.stop_walk.assert_called()
  assert "mining" in nav.lower()


def test_mining_done_when_label_gone(monkeypatch):
  brain = _ready_brain(confirm=3)
  brain._ore_peak = 0.99

  # Claramente abaixo de gone_threshold — fim real.
  gone = SimpleNamespace(found=False, raw_hit=False, score=0.2)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: gone)

  for i in range(2):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-wait-gone"
    assert brain.phase == Phase.READY_INTERACT
    assert f"{i + 1}/3" in nav
    assert "ore-weak" in nav

  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mining-done"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  brain.walker.clear_e_hold.assert_called_once()
  assert brain._mining_label_seen is False
  assert brain._mining_had_strong is False
  assert "SCAN" in nav or "restart" in nav.lower()


def test_ready_mining_hysteresis_keeps_hold_on_dip(monkeypatch):
  """0.99→0.98→0.92→0.95: dip com queda < drop — NÃO fim."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = _ready_brain(cfg, confirm=3)

  scores = [0.99, 0.98, 0.92, 0.95]
  for s in scores:
    strong = s >= 0.85
    hit = SimpleNamespace(found=strong, raw_hit=strong, score=s)
    monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx, h=hit: h)
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-hold", f"score={s} → {action} {nav}"
    assert brain.phase == Phase.READY_INTERACT

  brain.walker.clear_e_hold.assert_not_called()
  assert brain._ore_gone_streak == 0
  assert brain._ore_peak >= 0.99


def test_ready_mining_fp_plateau_above_match_uses_timeout(monkeypatch):
  """
  Logs: label gone mas score ~0.88–0.91 (found=false, ≥match).
  score≥match NÃO é ore-gone (dip mid-mine); hard timeout solta E.
  """
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["mine_hold_timeout_s"] = 2.0
  cfg["navigation"]["mining_ore"]["mine_hold_below_match_s"] = 0
  brain = _ready_brain(cfg, confirm=5)
  # _ready_brain leaves ready_at=0; set a fresh hold start for mid-hold checks.
  brain._mining_ready_at = time.perf_counter()

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.99)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-hold"

  # FP pós-gone: above match, no contrast — hold, não ore-gone.
  fp = SimpleNamespace(found=False, raw_hit=False, near_miss=True, score=0.88)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: fp)
  for i in range(5):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-hold", f"i={i} {action} {nav}"

  brain._mining_ready_at = 0.0  # force hard timeout
  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mining-done"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  brain.walker.clear_e_hold.assert_called_once()


def test_ready_mining_near_match_dip_keeps_hold(monkeypatch):
  """Sessão 15-25: peak 0.97 → ore 0.855 (≥match) NÃO deve ore-gone."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["gone_drop_from_peak"] = 0.10
  brain = _ready_brain(cfg, confirm=5)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.971)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-hold"

  dip = SimpleNamespace(found=False, raw_hit=False, near_miss=True, score=0.855)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: dip)
  for i in range(8):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-hold", f"i={i} {action} {nav}"
    assert brain.phase == Phase.READY_INTERACT
  brain.walker.clear_e_hold.assert_not_called()


def test_ready_mining_hold_timeout_forces_finish(monkeypatch):
  """FP platô ≥match sem drop suficiente → hard timeout solta E."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["mine_hold_timeout_s"] = 2.0
  cfg["navigation"]["mining_ore"]["mine_hold_below_match_s"] = 0
  brain = _ready_brain(cfg, confirm=5)
  brain._mining_ready_at = 0.0  # epoch → elapsed >> timeout

  # Platô sem drop (peak acompanhou score)
  fp = SimpleNamespace(found=False, raw_hit=False, near_miss=True, score=0.89)
  brain._ore_peak = 0.89
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: fp)
  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mining-done", f"{action} {nav}"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  assert (
    "timeout" in nav.lower()
    or "hold-timeout" in nav.lower()
    or "SCAN" in nav
    or "restart" in nav.lower()
  )
  brain.walker.clear_e_hold.assert_called_once()


def test_ready_mining_below_match_soft_timeout(monkeypatch):
  """Após N s com score < hold_min × confirm → force end (banda cinza)."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.75
  cfg["navigation"]["mining_ore"]["gone_threshold"] = 0.60
  cfg["navigation"]["mining_ore"]["gone_drop_from_peak"] = 0.20
  cfg["navigation"]["mining_ore"]["mine_hold_timeout_s"] = 0
  cfg["navigation"]["mining_ore"]["mine_hold_below_match_s"] = 1.0
  brain = _ready_brain(cfg, confirm=3)
  brain._mining_ready_at = 0.0
  brain._ore_peak = 0.90

  # 0.72: < hold_min, ≥ gone, drop 0.90−0.20=0.70 → 0.72 not < 0.70
  # → not absent via drop; soft timeout is the path.
  weak = SimpleNamespace(found=False, raw_hit=False, score=0.72)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: weak)
  actions = []
  for _ in range(5):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    actions.append(action)
    if action == "mining-done":
      break
  assert "mining-done" in actions
  brain.walker.clear_e_hold.assert_called()


def test_ready_mining_noise_after_gone_finishes(monkeypatch):
  """Após pico ~0.99, ruído pós-label ~0.50 (< gone 0.60) × N → finish."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = _ready_brain(cfg, confirm=5)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.99)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-hold"

  noise = SimpleNamespace(found=False, raw_hit=False, score=0.50)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: noise)
  for i in range(4):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-wait-gone", f"i={i} {action} {nav}"
  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mining-done"
  assert brain.phase == Phase.SCAN
  assert brain._allow_auto_lock is True
  brain.walker.clear_e_hold.assert_called_once()


def test_ready_mining_mid_noise_finishes(monkeypatch):
  """0.99 then 0.50×6 → finish (abaixo de gone_threshold)."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = _ready_brain(cfg, confirm=6)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.99)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  brain._tick_ready_mining(_ctx(), active=True)

  mid = SimpleNamespace(found=False, raw_hit=False, score=0.50)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: mid)
  for _ in range(5):
    assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-wait-gone"
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mining-done"


def test_ready_mining_low_noise_finishes(monkeypatch):
  """0.99 then 0.40×6 → finish."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = _ready_brain(cfg, confirm=6)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.99)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  brain._tick_ready_mining(_ctx(), active=True)

  low = SimpleNamespace(found=False, raw_hit=False, score=0.40)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: low)
  for _ in range(5):
    assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-wait-gone"
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mining-done"


def test_ready_mining_drop_from_peak_absent(monkeypatch):
  """score 0.65×N com peak 0.99: mediana < hold_min e queda ≥ drop → absent."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.75
  cfg["navigation"]["mining_ore"]["gone_threshold"] = 0.60
  cfg["navigation"]["mining_ore"]["gone_drop_from_peak"] = 0.20
  cfg["navigation"]["mining_ore"]["score_smooth_frames"] = 5
  brain = _ready_brain(cfg, confirm=3)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.99)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  brain._tick_ready_mining(_ctx(), active=True)

  drop = SimpleNamespace(found=False, raw_hit=False, score=0.65)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: drop)
  # Primeiros frames: mediana ainda alta (pico recente) → hold.
  # Depois mediana cai abaixo de hold_min → soft drop ausenta.
  saw_wait = False
  for i in range(8):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    if action == "mine-wait-gone":
      saw_wait = True
      break
    assert action == "mine-hold", f"i={i} {action} {nav}"
  assert saw_wait, "expected soft-drop absent after median fell"
  # confirm-1 more waits then done (already 1 wait above if break)
  for i in range(1):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-wait-gone", f"i={i} {action} {nav}"
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mining-done"


def test_ready_mining_shake_dip_0_84_keeps_hold(monkeypatch):
  """Bug: peak 0.95 → 0.84×10 (barra ainda visível) NÃO deve soltar E."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.75
  cfg["navigation"]["mining_ore"]["gone_threshold"] = 0.60
  cfg["navigation"]["mining_ore"]["gone_drop_from_peak"] = 0.20
  brain = _ready_brain(cfg, confirm=8)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.952)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-hold"

  dip = SimpleNamespace(found=False, raw_hit=False, near_miss=True, score=0.844)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: dip)
  for i in range(10):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-hold", f"i={i} {action} {nav}"
    assert brain.phase == Phase.READY_INTERACT
  brain.walker.clear_e_hold.assert_not_called()


def test_ready_mining_low_score_sustained_finishes(monkeypatch):
  """peak 0.95 then 0.40×10 → finish (label realmente gone)."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.75
  cfg["navigation"]["mining_ore"]["gone_threshold"] = 0.60
  brain = _ready_brain(cfg, confirm=8)

  strong = SimpleNamespace(found=True, raw_hit=True, score=0.95)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: strong)
  assert brain._tick_ready_mining(_ctx(), active=True)[0] == "mine-hold"

  low = SimpleNamespace(found=False, raw_hit=False, score=0.40)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: low)
  for i in range(7):
    action, nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "mine-wait-gone", f"i={i} {action} {nav}"
  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mining-done", f"{action} {nav}"
  brain.walker.clear_e_hold.assert_called_once()


def test_ready_mining_camera_shake_pattern_keeps_hold(monkeypatch):
  """Shake 0.92,0.80,0.91,0.83 (e repetido) → keep E."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  cfg["navigation"]["mining_ore"]["hold_min"] = 0.75
  cfg["navigation"]["mining_ore"]["gone_threshold"] = 0.60
  cfg["navigation"]["mining_ore"]["gone_drop_from_peak"] = 0.20
  brain = _ready_brain(cfg, confirm=8)

  pattern = [0.92, 0.80, 0.91, 0.83]
  for _ in range(3):
    for s in pattern:
      hit = SimpleNamespace(found=False, raw_hit=False, near_miss=True, score=s)
      monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx, h=hit: h)
      action, nav = brain._tick_ready_mining(_ctx(), active=True)
      assert action == "mine-hold", f"score={s} → {action} {nav}"
      assert brain.phase == Phase.READY_INTERACT
  brain.walker.clear_e_hold.assert_not_called()


def test_ready_mining_keeps_e_while_label_present(monkeypatch):
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.READY_INTERACT
  brain._mining_label_seen = True
  brain._mining_had_strong = True
  brain.walker.e_held_for_mine = True
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.clear_e_hold = MagicMock()

  hit = SimpleNamespace(found=True, raw_hit=True, score=0.97)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mine-hold"
  assert brain.phase == Phase.READY_INTERACT
  brain.walker.keep_e_for_mine.assert_called()
  brain.walker.clear_e_hold.assert_not_called()
  assert "mining" in nav.lower()


def test_ready_mine_hold_represses_e_when_up(monkeypatch):
  """
  Bug: READY + ore forte mas e_held_for_mine=False → mine-hold sem keydown E.
  Todo tick com ore presente deve begin+keep (forçar E down).
  """
  brain = Brain(_cfg(), node_detector=MagicMock())
  brain.phase = Phase.READY_INTERACT
  brain._mining_label_seen = True
  brain._mining_had_strong = True
  brain._ore_peak = 0.9
  brain.pursuit.stop_walk = MagicMock(return_value="stop")
  brain.walker.e_held_for_mine = False
  brain.walker.begin_probe_e = MagicMock(return_value="probe-e-down")
  brain.walker.keep_e_for_mine = MagicMock(
    side_effect=lambda: setattr(brain.walker, "e_held_for_mine", True)
  )
  brain.walker.clear_e_hold = MagicMock()

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=True, score=0.88
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_ready_mining(_ctx(), active=True)
  assert action == "mine-hold"
  assert brain.phase == Phase.READY_INTERACT
  assert brain.walker.e_held_for_mine is True
  brain.walker.begin_probe_e.assert_called_once()
  brain.walker.keep_e_for_mine.assert_called()
  brain.walker.clear_e_hold.assert_not_called()
  brain.pursuit.stop_walk.assert_called()
  assert "mining" in nav.lower()


def test_commit_holding_e_ensures_e_before_stop(monkeypatch):
  """
  holding_e + score≥thr: stop_walk com flag False soltava E via release_all.
  Commit deve setar E/flag ANTES de stop.
  """
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["match_threshold"] = 0.85
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.FINAL_APPROACH
  brain._announced_final = True
  brain._final_started_at = 0.0
  brain._pulse_count = 1
  brain._fa_sub = "holding_e"
  brain._fa_sub_at = 0.0
  brain.walker.e_held_for_mine = False
  order: list[str] = []

  def _begin():
    order.append("begin")
    return "probe-e-down"

  def _keep():
    order.append("keep")
    brain.walker.e_held_for_mine = True

  def _stop():
    order.append("stop")
    # Se flag ainda False aqui, regressão (release_all mataria E).
    assert brain.walker.e_held_for_mine is True
    return "stop"

  brain.walker.begin_probe_e = MagicMock(side_effect=_begin)
  brain.walker.keep_e_for_mine = MagicMock(side_effect=_keep)
  brain.pursuit.stop_walk = MagicMock(side_effect=_stop)
  brain.pursuit.mark_done = MagicMock()
  brain.pursuit.clear_stuck_blacklist = MagicMock()

  hit = SimpleNamespace(
    found=False, raw_hit=False, near_miss=True, score=0.90
  )
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: hit)

  action, nav = brain._tick_final_approach(
    _ctx(), active=True, pursuit=SimpleNamespace()
  )
  assert action == "ready-interact"
  assert brain.phase == Phase.READY_INTERACT
  assert order[0] == "begin"
  assert order[1] == "keep"
  assert "stop" in order
  assert order.index("keep") < order.index("stop")
  assert "READY" in nav


def test_ready_mining_no_finish_without_strong_hit(monkeypatch):
  """Sem hit forte prévio: score baixo não termina mine."""
  cfg = _cfg()
  cfg["navigation"]["mining_ore"]["gone_confirm_frames"] = 2
  cfg["navigation"]["mining_ore"]["gone_threshold"] = 0.70
  brain = Brain(cfg, node_detector=MagicMock())
  brain.phase = Phase.READY_INTERACT
  brain._mining_label_seen = False
  brain._mining_had_strong = False
  brain.walker.clear_e_hold = MagicMock()

  weak = SimpleNamespace(found=False, raw_hit=False, score=0.1)
  monkeypatch.setattr(brain, "_detect_mining_ore", lambda _ctx: weak)
  for _ in range(5):
    action, _nav = brain._tick_ready_mining(_ctx(), active=True)
    assert action == "idle"
    assert brain.phase == Phase.READY_INTERACT
  brain.walker.clear_e_hold.assert_not_called()
