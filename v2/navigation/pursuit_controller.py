"""
Navegação v2 — stack comprovado do bot.py (modo blip).

- Lock/track: node_detector.make_target_lock + track_target (v1)
- Rumo: screen_turn_error + walk_heading_from_arrow (desempate)
- Controle: MiningWalker / VehicleController (pulsos proporcionais)
- Facing estável: stable_facing_deg
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

from bot import resolve_heading_error, smooth_heading_error, stable_facing_deg
from logger import mlog
from navigator import camera_heading_from_arrow, normalize_angle_deg
from node_detector import MiningNode, MiningNodeDetector, NodeScanResult, TargetLock, bearing_deg
from progress_nav import arrow_to_target_with_dot
from visual_nav import VisualPursuitNavigator
from v2.navigation.bearing import (
  camera_align_bearing,
  target_ahead_dot_facing,
)
from v2.navigation.camera_walker import CameraAlignController
from v2.navigation.turn_coordinator import TurnCoordinator
from walker import MiningWalker

from v2.core.types import TargetLock as V2TargetLock


@dataclass
class PursuitOutput:
  target: MiningNode | None
  bearing_deg: float | None
  target_dot: float | None
  dist_px: float
  aligned: bool
  arrived: bool
  nav_status: str
  display_lock: V2TargetLock | None
  move_phase: str = "idle"


class PursuitController:
  def __init__(self, cfg: dict[str, Any], detector: MiningNodeDetector) -> None:
    self.detector = detector
    nav = cfg.get("navigation", {})
    pursuit = cfg.get("pursuit", {})

    self.arrive_px = float(nav.get("arrive_px", pursuit.get("arrive_px", 15)))
    self.min_pick_px = float(nav.get("min_pick_px", pursuit.get("min_pick_px", 16)))
    self.lost_px = float(cfg.get("lost_blip_distance_px", 22))
    self.chain_px = float(
      cfg.get("target_lock_chain_px", pursuit.get("lock_chain_px", 55))
    )
    self.close_hold_px = float(
      nav.get("close_hold_px", pursuit.get("close_hold_px", 34))
    )
    # Sticky: match por jump curto / last-live; sem nó virtual.
    self.sticky_enter_px = float(nav.get("sticky_enter_px", 40.0))
    # Se blip real reaparece claramente a frente no hold, cancela e retoma.
    self.hold_recover_px = float(nav.get("hold_recover_px", 14.0))
    self.sticky_live_reacquire_px = float(nav.get("sticky_live_reacquire_px", 22.0))
    # Ghost lock (sem rematch live): longe → drop rápido e re-SCAN; perto → grace maior.
    self.lock_max_lost_frames = max(
      3, int(nav.get("lock_max_lost_frames", 15))
    )
    self.lock_max_lost_frames_close = max(
      self.lock_max_lost_frames,
      int(nav.get("lock_max_lost_frames_close", 45)),
    )
    # Após ghost abandon: evita re-lock do mesmo XY por TTL (anti thrash lock/lose).
    # Não usa `_done_xy` permanente — flicker real podia blacklistar o nó bom.
    self.ghost_avoid_ttl_s = float(nav.get("ghost_avoid_ttl_s", 5.0))
    self.ghost_avoid_radius_px = float(nav.get("ghost_avoid_radius_px", 28.0))
    # STUCK/IDLE: force-avoid do nó obstruído (TTL longo + raio maior que ghost).
    # Polar (dist + bearing rel. facing) sobrevive à rotação do minimapa no SCAN_SPIN.
    self.stuck_avoid_ttl_s = float(nav.get("stuck_avoid_ttl_s", 45.0))
    self.stuck_avoid_radius_px = float(nav.get("stuck_avoid_radius_px", 40.0))
    self.stuck_avoid_dist_tol_px = float(nav.get("stuck_avoid_dist_tol_px", 22.0))
    self.stuck_avoid_bearing_tol_deg = float(
      nav.get("stuck_avoid_bearing_tol_deg", 40.0)
    )
    # mark_done / _done_xy: raio de exclusão (minados + stuck; stuck limpável).
    self.done_radius_px = float(nav.get("done_radius_px", 36.0))
    # lock_nearest: rejeita fagulha/estrada (área minúscula / pouca circularidade).
    self.lock_min_circularity = float(nav.get("lock_min_circularity", 0.55))
    self.lock_min_area = float(nav.get("lock_min_area", 8.0))
    self.stagnation_s = float(cfg.get("target_stagnation_s", 15.0))
    self.heading_smooth_alpha = float(cfg.get("heading_error_smoothing", 0.28))
    self.camera_bearing_max_jump_deg = float(
      nav.get("camera", {}).get("bearing_max_jump_deg", 45.0)
    )
    self.camera_pin_dist_slop_px = float(nav.get("pin_dist_px", 22))
    self.camera_pin_angle_slop_deg = float(
      nav.get("camera", {}).get("pin_angle_slop_deg", 55.0)
    )
    self.camera_align_only = bool(nav.get("camera", {}).get("align_only", False))
    self.first_person = bool(nav.get("camera", {}).get("first_person", True))
    if str(nav.get("control_mode", "camera")).lower() == "camera":
      self.first_person = True
    cam = nav.get("camera", {})
    self.fine_align_deadband_deg = float(cam.get("fine_align_deadband_deg", 1.5))
    self.fine_align_settle_frames = max(1, int(cam.get("fine_align_settle_frames", 2)))
    # Pós fine-align: W até close_walk_px de progresso blip→pivot (não confiar em dist=0).
    # Pixels = distância euclidiana na imagem do minimapa entre centro do blip e pivô do jogador.
    self.close_walk_px = float(
      nav.get("close_walk_px", pursuit.get("close_walk_px", 15.5))
    )
    # Teto absoluto de W (também quando o blip ainda "existe" mas progresso congelou).
    self.close_walk_max_s = float(
      nav.get(
        "close_walk_max_s",
        pursuit.get(
          "close_walk_max_s",
          nav.get("close_walk_s", pursuit.get("close_walk_s", 2.8)),
        ),
      )
    )
    # Rejeita sep colapsada na seta (HUD/dist zeroed) — não contar como progresso.
    self.close_walk_min_sep_px = float(
      nav.get("close_walk_min_sep_px", pursuit.get("close_walk_min_sep_px", 6.0))
    )
    # Sem avanço de progresso por este tempo → estagnação (para ou estima).
    self.close_walk_stall_s = float(
      nav.get("close_walk_stall_s", pursuit.get("close_walk_stall_s", 0.45))
    )
    # Frames sem blip live após já ter andado → para (evita W até max_s).
    self.close_walk_lost_frames = max(
      1,
      int(nav.get("close_walk_lost_frames", pursuit.get("close_walk_lost_frames", 4))),
    )
    # Taxa mínima (px/s) para estimar tempo restante se o blip sumir.
    self.close_walk_min_rate_px_s = float(
      nav.get(
        "close_walk_min_rate_px_s",
        pursuit.get("close_walk_min_rate_px_s", 8.0),
      )
    )
    # GOTO walk: sem progresso de dist → path blocked.
    # Até stuck_d_max_attempts: pulse D (stuck_d_hold_ms) e retoma walk.
    # Só então mark_stuck + SCAN. 0 em stuck_idle_s desliga. Só fora de
    # fine_align / close_walk / final / READY.
    # `_stuck_d_attempts` é per-lock (não zera em jitter de dist).
    # stuck_min_dist default = arrive_px: stuck while not yet arrived (dist > arrive).
    # Do NOT use arrive+10 — that disables stuck in the near-but-blocked band (e.g. dist=19).
    self.stuck_idle_s = float(nav.get("stuck_idle_s", pursuit.get("stuck_idle_s", 2.0)))
    self.stuck_min_dist_px = float(
      nav.get(
        "stuck_min_dist_px",
        pursuit.get("stuck_min_dist_px", self.arrive_px),
      )
    )
    self.stuck_progress_px = float(
      nav.get("stuck_progress_px", pursuit.get("stuck_progress_px", 1.5))
    )
    self.stuck_d_hold_ms = float(
      nav.get("stuck_d_hold_ms", pursuit.get("stuck_d_hold_ms", 2000.0))
    )
    self.stuck_d_max_attempts = max(
      0,
      int(nav.get("stuck_d_max_attempts", pursuit.get("stuck_d_max_attempts", 3))),
    )
    cam_cfg = nav.get("camera", {})
    self.stuck_align_deg = float(
      nav.get(
        "stuck_align_deg",
        cam_cfg.get("walk_max_deg", nav.get("align_deg", 12.0)),
      )
    )
    self.final_pulse_max = max(1, int(nav.get("final_pulse_max", 5)))
    self.final_approach_timeout_s = float(nav.get("final_approach_timeout_s", 0.0))
    self.facing_max_jump_deg = float(cfg.get("facing_max_jump_deg", 22.0))
    self.min_target_distance_px = float(
      cfg.get("min_target_distance_px", self.min_pick_px)
    )

    self.visual_nav = VisualPursuitNavigator(
      align_deg=float(nav.get("align_deg", pursuit.get("align_deg", 10))),
      min_closing_px=float(cfg.get("visual_nav", {}).get("min_closing_px", 0.4)),
    )
    self.turn = TurnCoordinator(
      align_deg=float(nav.get("align_deg", pursuit.get("align_deg", 12))),
      settle_frames=int(nav.get("settle_frames", 4)),
      commit_ms=float(nav.get("turn_commit_ms", 320)),
    )
    control_mode = str(nav.get("control_mode", "camera")).lower()
    self.control_mode = control_mode
    if control_mode == "camera":
      self.walker = CameraAlignController(cfg)
    else:
      self.walker = MiningWalker(cfg.get("walker", {}))

    self._lock: TargetLock | None = None
    self._smooth_heading: float | None = None
    self._smooth_facing: float | None = None
    self._approach_min_dist: float | None = None
    self._stagnant_since: float | None = None
    self._display_id = 1
    self._done_xy: list[tuple[float, float]] = []
    # Subset de `_done_xy` originado de STUCK/IDLE (limpável no mine success).
    self._stuck_done_xy: list[tuple[float, float]] = []
    # (x, y, until_perf) — soft-avoid temporário pós ghost abandon.
    self._ghost_avoid_xy: list[tuple[float, float, float]] = []
    # (x, y, until_perf) — soft-avoid longo pós STUCK/IDLE (append; multi-nó).
    self._stuck_avoid_xy: list[tuple[float, float, float]] = []
    # (dist_px, bearing_rel_facing_deg, until_perf) — anti re-lock após spin.
    self._stuck_polar_avoid: list[tuple[float, float, float]] = []
    self._prev_abs_bearing: float | None = None
    self._hold_for_mine = False
    self._sticky_id = False
    self._last_live_x: float | None = None
    self._last_live_y: float | None = None
    self._fine_aligning = False
    self._fine_aligned = False
    self._fine_align_ok_frames = 0
    self._close_walking = False
    self._close_walk_done = False
    self._close_walk_started_at: float | None = None
    self._close_walk_start_sep: float | None = None
    self._close_walk_progress: float = 0.0
    self._close_walk_blip_lost = False
    self._close_walk_last_progress_at: float | None = None
    self._close_walk_px_per_s: float = 0.0
    self._close_walk_lost_count: int = 0
    self._close_walk_budget_s: float | None = None
    self._final_approaching = False
    self._stuck_since: float | None = None
    self._stuck_best_dist: float | None = None
    # Per-lock: D-strafe recoveries before blacklist/switch (0..max).
    self._stuck_d_attempts: int = 0
    # Após pulse D: obriga realign de câmera (sem W) até |brg| ≤ stuck_align_deg.
    self._need_post_stuck_realign: bool = False

  _STUCK_SKIP_PHASES = frozenset(
    {"fine_align", "close_walk", "close_done", "final_approach", "aligned"}
  )

  @property
  def v1_lock(self) -> TargetLock | None:
    return self._lock

  def _clear_live_track(self) -> None:
    self._last_live_x = None
    self._last_live_y = None

  def _reset_stuck_idle(self) -> None:
    self._stuck_since = None
    self._stuck_best_dist = None

  def _reset_stuck_d_attempts(self) -> None:
    self._stuck_d_attempts = 0
    self._need_post_stuck_realign = False

  def _clear_close_walk(self) -> None:
    self._close_walking = False
    self._close_walk_done = False
    self._close_walk_started_at = None
    self._close_walk_start_sep = None
    self._close_walk_progress = 0.0
    self._close_walk_blip_lost = False
    self._close_walk_last_progress_at = None
    self._close_walk_px_per_s = 0.0
    self._close_walk_lost_count = 0
    self._close_walk_budget_s = None
    self._final_approaching = False

  def _clear_fine_align(self) -> None:
    self._fine_aligning = False
    self._fine_aligned = False
    self._fine_align_ok_frames = 0
    self._clear_close_walk()

  def reset(self) -> None:
    self._lock = None
    self._smooth_heading = None
    self._smooth_facing = None
    self._approach_min_dist = None
    self._stagnant_since = None
    self._hold_for_mine = False
    self._sticky_id = False
    self._clear_live_track()
    self._clear_fine_align()
    self._reset_stuck_idle()
    self._reset_stuck_d_attempts()
    self.walker.stop()
    self.visual_nav.reset()
    self.turn.reset()

  def mark_done(self, *, from_stuck: bool = False) -> None:
    if self._lock is not None:
      xy = (float(self._lock.locked_x), float(self._lock.locked_y))
      self._done_xy.append(xy)
      if from_stuck:
        self._stuck_done_xy.append(xy)
    self._lock = None
    self._approach_min_dist = None
    self._stagnant_since = None
    self._smooth_heading = None
    self._hold_for_mine = False
    self._sticky_id = False
    self._clear_live_track()
    self._clear_fine_align()
    self._reset_stuck_idle()
    self._reset_stuck_d_attempts()
    self.visual_nav.reset()
    self.turn.reset()

  def mark_stuck(self, *, facing_deg: float | None = None) -> None:
    """
    Path blocked (STUCK/IDLE): force-avoid o XY do lock atual (append).

    Cada stuck acrescenta entradas — A depois B → ambos evitados até
    `clear_stuck_blacklist()` (mine success).

    - `_done_xy` + `_stuck_done_xy` (sessão, limpável)
    - soft-avoid longo (`stuck_avoid_ttl_s`)
    - polar (dist + bearing rel. facing) para não re-travar o mesmo nó
      depois que o SCAN_SPIN gira o minimapa e o XY absoluto muda
    """
    if self._lock is not None:
      lx = float(self._lock.locked_x)
      ly = float(self._lock.locked_y)
      ax = (
        float(self._last_live_x) if self._last_live_x is not None else lx
      )
      ay = (
        float(self._last_live_y) if self._last_live_y is not None else ly
      )
      # Live + locked (se divergiram) — cobre pin/jitter.
      if math.hypot(ax - lx, ay - ly) > 5.0:
        self._done_xy.append((ax, ay))
        self._stuck_done_xy.append((ax, ay))
        self._push_stuck_avoid(ax, ay)
        self._push_stuck_avoid(lx, ly)
      else:
        self._push_stuck_avoid(ax, ay)
      dist = float(getattr(self._lock, "last_distance_px", 0.0) or 0.0)
      brg = getattr(self._lock, "last_bearing_deg", None)
      face = facing_deg if facing_deg is not None else self._smooth_facing
      if brg is not None and face is not None and dist > 0:
        rel = normalize_angle_deg(float(brg) - float(face))
        self._push_stuck_polar(dist, rel)
      mlog(
        f"[v2] STUCK avoid — xy=({ax:.0f},{ay:.0f}) "
        f"ttl={self.stuck_avoid_ttl_s:.0f}s r={self.stuck_avoid_radius_px:.0f}px "
        f"n={len(self._stuck_done_xy) + 1}"
      )
    self.mark_done(from_stuck=True)

  def clear_stuck_blacklist(self) -> None:
    """
    Limpa evitações de STUCK/IDLE após mine success.

    Remove só entradas stuck de `_done_xy` (nós minados com sucesso ficam).
    Soft-avoid XY + polar também zeram. Não chamar em SCAN aleatório.
    """
    n_done = len(self._stuck_done_xy)
    n_soft = len(self._stuck_avoid_xy)
    n_polar = len(self._stuck_polar_avoid)
    if n_done == 0 and n_soft == 0 and n_polar == 0:
      return
    if self._stuck_done_xy:
      stuck = set(self._stuck_done_xy)
      self._done_xy = [xy for xy in self._done_xy if xy not in stuck]
      self._stuck_done_xy.clear()
    self._stuck_avoid_xy.clear()
    self._stuck_polar_avoid.clear()
    mlog(
      f"[v2] STUCK blacklist cleared — "
      f"done={n_done} soft={n_soft} polar={n_polar}"
    )

  def _is_done(self, x: float, y: float) -> bool:
    rad = float(self.done_radius_px)
    for dx, dy in self._done_xy:
      if math.hypot(x - dx, y - dy) < rad:
        return True
    return False

  def _prune_ghost_avoid(self) -> None:
    now = time.perf_counter()
    self._ghost_avoid_xy = [
      (x, y, until)
      for x, y, until in self._ghost_avoid_xy
      if until > now
    ]

  def _push_ghost_avoid(self, x: float, y: float) -> None:
    """Soft-blacklist temporário do XY abandonado (não permanente)."""
    ttl = float(self.ghost_avoid_ttl_s)
    if ttl <= 0:
      return
    self._prune_ghost_avoid()
    until = time.perf_counter() + ttl
    self._ghost_avoid_xy.append((float(x), float(y), until))

  def _is_ghost_avoided(self, x: float, y: float) -> bool:
    self._prune_ghost_avoid()
    rad = float(self.ghost_avoid_radius_px)
    for ax, ay, _until in self._ghost_avoid_xy:
      if math.hypot(x - ax, y - ay) < rad:
        return True
    return False

  def _prune_stuck_avoid(self) -> None:
    now = time.perf_counter()
    self._stuck_avoid_xy = [
      (x, y, until)
      for x, y, until in self._stuck_avoid_xy
      if until > now
    ]
    self._stuck_polar_avoid = [
      (d, rel, until)
      for d, rel, until in self._stuck_polar_avoid
      if until > now
    ]

  def _push_stuck_avoid(self, x: float, y: float) -> None:
    """Soft-blacklist longo do XY que causou STUCK/IDLE."""
    ttl = float(self.stuck_avoid_ttl_s)
    if ttl <= 0:
      return
    self._prune_stuck_avoid()
    until = time.perf_counter() + ttl
    self._stuck_avoid_xy.append((float(x), float(y), until))

  def _push_stuck_polar(self, dist_px: float, bearing_rel_deg: float) -> None:
    ttl = float(self.stuck_avoid_ttl_s)
    if ttl <= 0:
      return
    self._prune_stuck_avoid()
    until = time.perf_counter() + ttl
    self._stuck_polar_avoid.append(
      (float(dist_px), float(bearing_rel_deg), until)
    )

  def _is_stuck_avoided(
    self,
    x: float,
    y: float,
    *,
    player_x: float | None = None,
    player_y: float | None = None,
    facing_deg: float | None = None,
    distance_px: float | None = None,
  ) -> bool:
    """True se XY soft-avoided ou mesmo nó polar (dist+bearing rel. facing)."""
    self._prune_stuck_avoid()
    rad = float(self.stuck_avoid_radius_px)
    for ax, ay, _until in self._stuck_avoid_xy:
      if math.hypot(x - ax, y - ay) < rad:
        return True
    if (
      facing_deg is None
      or player_x is None
      or player_y is None
      or not self._stuck_polar_avoid
    ):
      return False
    dist = (
      float(distance_px)
      if distance_px is not None
      else math.hypot(x - player_x, y - player_y)
    )
    brg = bearing_deg(player_x, player_y, x, y)
    rel = normalize_angle_deg(brg - float(facing_deg))
    d_tol = float(self.stuck_avoid_dist_tol_px)
    b_tol = float(self.stuck_avoid_bearing_tol_deg)
    for adist, arel, _until in self._stuck_polar_avoid:
      if abs(dist - adist) <= d_tol and abs(
        normalize_angle_deg(rel - arel)
      ) <= b_tol:
        return True
    return False

  def _in_close_protect(self) -> bool:
    """Fine/close/final/hold — não abandonar ghost (Mining ore depende do hold)."""
    return bool(
      self._fine_aligning
      or self._fine_aligned
      or self._close_walking
      or self._final_approaching
      or self._hold_for_mine
    )

  def _ghost_lost_exceeded(self, lock: TargetLock) -> bool:
    """True se lock fantasma ficou sem rematch live tempo demais (só longe)."""
    if self._in_close_protect():
      return False
    lost = int(getattr(lock, "lost_frames", 0) or 0)
    dist = float(getattr(lock, "last_distance_px", 999.0) or 999.0)
    closeish = self._sticky_id or dist <= float(self.sticky_enter_px)
    limit = (
      self.lock_max_lost_frames_close if closeish else self.lock_max_lost_frames
    )
    return lost > limit

  def _abandon_ghost_lock(self) -> None:
    """
    Limpa lock fantasma sem mark_done permanente.

    Soft-avoid temporário do XY (`ghost_avoid_ttl_s`) impede re-lock imediato
    do mesmo flicker; cooldown de re-lock + spin-once ficam no Brain.
    """
    if self._lock is not None:
      dist = float(getattr(self._lock, "last_distance_px", 0.0) or 0.0)
      lost = int(getattr(self._lock, "lost_frames", 0) or 0)
      ax = float(self._lock.locked_x)
      ay = float(self._lock.locked_y)
      mlog(
        f"[v2] ghost abandon — clear lock "
        f"(dist={dist:.0f}px lost={lost})"
      )
      self._push_ghost_avoid(ax, ay)
    self._lock = None
    self._approach_min_dist = None
    self._stagnant_since = None
    self._smooth_heading = None
    self._hold_for_mine = False
    self._sticky_id = False
    self._clear_live_track()
    self._clear_fine_align()
    self._reset_stuck_idle()
    self._reset_stuck_d_attempts()
    self.visual_nav.reset()
    self.turn.reset()

  def check_stuck_idle(
    self,
    dist_px: float,
    *,
    bearing_deg: float | None,
    move_phase: str,
    expecting_walk: bool,
  ) -> bool:
    """
    True se, durante GOTO walk alinhado, dist não caiu o bastante por stuck_idle_s.

    Não dispara em fine_align / close_walk / final_approach / aligned (paradas
    intencionais). Também ignora realign (|brg| > stuck_align_deg).

    stuck_min_dist_px (default = arrive_px): só rastreia stuck enquanto
    dist > stuck_min (ainda não chegou). Congelado em dist=19 com W alinhado
    dispara após stuck_idle_s.
    """
    if self.stuck_idle_s <= 0:
      self._reset_stuck_idle()
      return False
    phase = str(move_phase or "")
    if phase in self._STUCK_SKIP_PHASES or self._in_close_protect():
      self._reset_stuck_idle()
      return False
    if not expecting_walk or bearing_deg is None:
      self._reset_stuck_idle()
      return False
    if abs(float(bearing_deg)) > float(self.stuck_align_deg):
      self._reset_stuck_idle()
      return False
    # Not yet arrived / still outside arrive band → allow stuck.
    if float(dist_px) <= float(self.stuck_min_dist_px):
      self._reset_stuck_idle()
      return False

    now = time.perf_counter()
    dist = float(dist_px)
    need = float(self.stuck_progress_px)
    if self._stuck_best_dist is None:
      self._stuck_best_dist = dist
      self._stuck_since = now
      return False
    if dist < self._stuck_best_dist - need:
      # Progress reinicia só o timer STUCK — NÃO zera `_stuck_d_attempts`.
      # Jitter/D-strafe (~1–3px) apagava o contador e o bot nunca chegava
      # a 3× D → mark_stuck no mesmo lock. Attempts só zeram em novo lock /
      # mark_done / reset.
      self._stuck_best_dist = dist
      self._stuck_since = now
      return False
    if self._stuck_since is None:
      self._stuck_since = now
      return False
    if now - self._stuck_since >= float(self.stuck_idle_s):
      self._reset_stuck_idle()
      return True
    return False

  def recover_stuck_d(self) -> str:
    """
    STUCK/IDLE recovery: solta W, segura D (stuck_d_hold_ms), solta D.

    Incrementa `_stuck_d_attempts`. Após o pulse, arma realign obrigatório
    (sem W) até |brg| ≤ stuck_align_deg — evita retomar walk de lado.
    Limpa `_smooth_heading` para o próximo frame não herdar rumo pré-strafe
    (clamp bearing_max_jump).
    Caller só invoca se attempts < max.
    """
    hold_ms = float(self.stuck_d_hold_ms)
    pulse = getattr(self.walker, "pulse_strafe_d", None)
    if callable(pulse):
      action = pulse(hold_ms=hold_ms)
    else:
      action = f"strafe-d-{hold_ms:.0f}ms"
    self._stuck_d_attempts = int(self._stuck_d_attempts) + 1
    self._reset_stuck_idle()
    # Não usar _reset_stuck_d_attempts (zera o contador).
    self._need_post_stuck_realign = True
    self._smooth_heading = None
    return str(action)

  def lock_nearest(
    self,
    scan: NodeScanResult,
    *,
    facing_deg: float | None = None,
  ) -> MiningNode | None:
    """
    Escolhe blip para lock inicial.

    Primário: menor distância ao jogador (minimap px). Hard rejects
    (tier, min_pick, circularidade, área, done) filtram fagulha/estrada.
    Facing à frente só desempata nós quase equidistantes.
    """
    candidates: list[MiningNode] = []
    for node in scan.nodes:
      if node.tier.lower() not in self.detector.allowed_tiers:
        continue
      if node.distance_px < self.min_pick_px:
        continue
      if float(node.circularity) < self.lock_min_circularity:
        continue
      if float(node.area) < self.lock_min_area:
        continue
      if self._is_done(node.x, node.y):
        continue
      if self._is_ghost_avoided(node.x, node.y):
        continue
      if self._is_stuck_avoided(
        node.x,
        node.y,
        player_x=float(scan.player_x),
        player_y=float(scan.player_y),
        facing_deg=facing_deg,
        distance_px=float(node.distance_px),
      ):
        continue
      candidates.append(node)
    if not candidates:
      return None

    def _score(node: MiningNode) -> float:
      # Distância domina; ahead ≤ ~3 px de desempate.
      score = float(node.distance_px)
      if facing_deg is not None:
        brg = bearing_deg(scan.player_x, scan.player_y, node.x, node.y)
        err = abs(normalize_angle_deg(brg - facing_deg))
        score += (err / 180.0) * 3.0
      return score

    pick = min(candidates, key=_score)
    self._lock = self.detector.make_target_lock(scan, pick)
    self._approach_min_dist = pick.distance_px
    self._stagnant_since = None
    self._smooth_heading = None
    self._hold_for_mine = False
    self._sticky_id = False
    self._clear_fine_align()
    self._reset_stuck_idle()
    self._reset_stuck_d_attempts()
    self._last_live_x = float(pick.x)
    self._last_live_y = float(pick.y)
    self.visual_nav.reset()
    return pick

  def lock_next(
    self,
    scan: NodeScanResult,
    *,
    facing_deg: float | None = None,
  ) -> MiningNode | None:
    self.mark_done()
    return self.lock_nearest(scan, facing_deg=facing_deg)

  def resolve_target(self, scan: NodeScanResult) -> MiningNode | None:
    if self._lock is None:
      return None
    if self.control_mode == "camera":
      return self._resolve_target_camera(scan)
    tracked, updated = self.detector.track_target(
      scan,
      self._lock,
      chain_radius_px=self.chain_px,
      close_hold_px=self.close_hold_px,
    )
    self._lock = updated
    if tracked is not None:
      return tracked
    return self.detector.node_from_lock(
      self._lock,
      player_x=scan.player_x,
      player_y=scan.player_y,
    )

  def _find_hold_recover_live(
    self,
    scan: NodeScanResult,
    *,
    min_ahead_px: float,
  ) -> MiningNode | None:
    """Blip real do mesmo tier ainda claramente a frente do pivô → retomar pursuit."""
    lock = self._lock
    if lock is None:
      return None
    px, py = scan.player_x, scan.player_y
    prev_a = float(lock.last_bearing_deg)
    reacq = float(self.sticky_live_reacquire_px)
    candidates: list[MiningNode] = []
    for n in self._live_tier_nodes(scan, tier=lock.tier):
      if n.distance_px <= min_ahead_px:
        continue
      jump_live = 999.0
      if self._last_live_x is not None and self._last_live_y is not None:
        jump_live = math.hypot(n.x - self._last_live_x, n.y - self._last_live_y)
      # Durante hold o lock pode estar no pivô — preferir last-live.
      near_track = jump_live <= reacq
      if not near_track and self._last_live_x is None:
        near_track = math.hypot(n.x - lock.locked_x, n.y - lock.locked_y) <= reacq
      if not near_track:
        continue
      ang = abs(normalize_angle_deg(bearing_deg(px, py, n.x, n.y) - prev_a))
      if ang <= 50.0:
        candidates.append(n)
    if not candidates:
      return None
    return min(
      candidates,
      key=lambda n: (
        math.hypot(
          n.x - (self._last_live_x if self._last_live_x is not None else lock.locked_x),
          n.y - (self._last_live_y if self._last_live_y is not None else lock.locked_y),
        ),
        n.distance_px,
      ),
    )

  def _live_tier_nodes(self, scan: NodeScanResult, *, tier: str) -> list[MiningNode]:
    """Blips live do tier: scan.nodes + track_nodes (unlock perto do pivô)."""
    allowed = {t.lower() for t in self.detector.allowed_tiers}
    tier_l = tier.lower()
    out: list[MiningNode] = []
    for pool in (scan.nodes, scan.track_nodes or []):
      for n in pool:
        if n.tier.lower() not in allowed:
          continue
        if n.tier.lower() != tier_l:
          continue
        if any(math.hypot(n.x - k.x, n.y - k.y) < 4.0 for k in out):
          continue
        out.append(n)
    return out

  def _resolve_target_camera(self, scan: NodeScanResult) -> MiningNode | None:
    """
    Longe: track real.
    Perto (sticky): rematch live por jump curto / last-live.
    Sem match: congela última posição live (sem fabricar nó virtual).
    """
    lock = self._lock
    assert lock is not None

    tier_nodes = self._live_tier_nodes(scan, tier=lock.tier)
    px, py = scan.player_x, scan.player_y
    ref_dist = float(lock.last_distance_px or lock.pick_distance_px)
    min_seen = float(getattr(lock, "min_seen_distance_px", ref_dist) or ref_dist)
    prev_a = float(lock.last_bearing_deg)
    locked_area = float(getattr(lock, "locked_area", 0.0) or 0.0)
    approach = (
      self._approach_min_dist
      if self._approach_min_dist is not None
      else min_seen
    )

    enter = float(self.sticky_enter_px)
    if approach <= enter or min_seen <= enter:
      self._sticky_id = True
    sticky = self._sticky_id or self._hold_for_mine

    def _node_angle(n: MiningNode) -> float:
      return bearing_deg(px, py, n.x, n.y)

    def _area_ratio(n: MiningNode) -> float:
      if locked_area <= 1.0 or n.area <= 1.0:
        return 1.0
      return max(n.area / locked_area, locked_area / n.area)

    def _jump_lock(n: MiningNode) -> float:
      return math.hypot(n.x - lock.locked_x, n.y - lock.locked_y)

    def _jump_last_live(n: MiningNode) -> float:
      if self._last_live_x is None or self._last_live_y is None:
        return 999.0
      return math.hypot(n.x - self._last_live_x, n.y - self._last_live_y)

    def _commit_live(best: MiningNode) -> MiningNode:
      lock.lost_frames = 0
      lock.locked_x = best.x
      lock.locked_y = best.y
      lock.last_distance_px = best.distance_px
      lock.last_bearing_deg = _node_angle(best)
      if best.area > 0:
        lock.locked_area = (
          best.area
          if locked_area <= 1.0
          else 0.8 * locked_area + 0.2 * best.area
        )
      if hasattr(lock, "min_seen_distance_px"):
        lock.min_seen_distance_px = min(
          float(lock.min_seen_distance_px or best.distance_px),
          best.distance_px,
        )
      self._last_live_x = float(best.x)
      self._last_live_y = float(best.y)
      self._lock = lock
      return best

    def _near_player_arrow(n: MiningNode) -> bool:
      # Recusar blob no pivô (seta do jogador), nunca travar nela.
      return n.distance_px < max(6.0, self.min_pick_px * 0.45)

    def _far_quality(n: MiningNode) -> bool:
      return (
        float(n.circularity) >= max(0.45, self.lock_min_circularity * 0.9)
        and float(n.area) >= max(4.0, self.lock_min_area * 0.75)
        and not _near_player_arrow(n)
      )

    def _far_quality_soft(n: MiningNode) -> bool:
      # Flicker HSV: aceita disco um pouco pior no rematch last-live.
      return (
        float(n.circularity) >= max(0.40, self.lock_min_circularity * 0.8)
        and float(n.area) >= max(3.0, self.lock_min_area * 0.55)
        and not _near_player_arrow(n)
      )

    def _commit_lost() -> MiningNode | None:
      # Congela última posição conhecida — sem puxar ao pivô.
      # Longe + sem rematch por N frames → tenta rescue live, senão abandona.
      lock.lost_frames += 1
      lock.last_distance_px = float(
        math.hypot(lock.locked_x - px, lock.locked_y - py)
      )
      lock.last_bearing_deg = bearing_deg(px, py, lock.locked_x, lock.locked_y)
      if hasattr(lock, "min_seen_distance_px"):
        lock.min_seen_distance_px = min(
          float(lock.min_seen_distance_px or lock.last_distance_px),
          lock.last_distance_px,
        )
      self._lock = lock
      if self._ghost_lost_exceeded(lock):
        # Disco gray ainda na tela (outro XY) → retarget em vez de SCAN thrash.
        rescue_pool = [
          n
          for n in tier_nodes
          if _far_quality(n)
          and not self._is_ghost_avoided(n.x, n.y)
          and not self._is_stuck_avoided(n.x, n.y)
        ]
        if rescue_pool:
          self._push_ghost_avoid(float(lock.locked_x), float(lock.locked_y))

          def _rescue_score(n: MiningNode) -> tuple[float, float]:
            return (_jump_last_live(n), float(n.distance_px))

          return _commit_live(min(rescue_pool, key=_rescue_score))
        self._abandon_ghost_lock()
        return None
      return self.detector.node_from_lock(lock, player_x=px, player_y=py)

    def _sticky_score(n: MiningNode) -> float:
      # Preferir centro do blip mais perto do último lock / last-live.
      return (
        _jump_lock(n) * 3.0
        + _jump_last_live(n) * 2.5
        + abs(normalize_angle_deg(_node_angle(n) - prev_a)) * 0.25
        + (0.0 if _area_ratio(n) <= 1.55 else 40.0)
      )

    # --- STICKY (live only) ---
    if sticky:
      # Perto / fine-align: rematch curto ao centro do gray; sem snap à seta.
      fine = self._fine_aligning or self._fine_aligned
      max_jump = 12.0 if (fine or approach <= 28.0) else 20.0
      reacq = float(self.sticky_live_reacquire_px)
      if fine:
        reacq = min(reacq, 16.0)
      pool = [n for n in tier_nodes if not _near_player_arrow(n)]
      same = [
        n
        for n in pool
        if _jump_lock(n) <= max_jump or _jump_last_live(n) <= reacq
      ]
      if same:
        return _commit_live(min(same, key=_sticky_score))
      # Rematch curto: nearest gray ao último lock (centro), raio reacq.
      near_lock = [n for n in pool if _jump_lock(n) <= reacq]
      if near_lock:
        return _commit_live(min(near_lock, key=_sticky_score))
      if self._last_live_x is not None and self._last_live_y is not None:
        near_live = [
          n
          for n in pool
          if _jump_last_live(n) <= reacq
          and abs(normalize_angle_deg(_node_angle(n) - prev_a)) <= 50.0
        ]
        if near_live:
          return _commit_live(min(near_live, key=_sticky_score))
      return _commit_lost()

    # --- Longe: track real (só discos com qualidade mínima — evita fagulha/estrada) ---
    def _score(n: MiningNode) -> float:
      jump = _jump_lock(n)
      ang_prev = abs(normalize_angle_deg(_node_angle(n) - prev_a))
      dist_delta = n.distance_px - ref_dist
      dist_err = abs(dist_delta) * 0.5 if dist_delta <= 8.0 else dist_delta * 5.0
      ar = _area_ratio(n)
      area_pen = 0.0 if ar <= 1.45 else 25.0 * (ar - 1.45)
      live_j = _jump_last_live(n)
      return jump * 2.5 + live_j * 1.2 + ang_prev * 1.5 + dist_err + area_pen

    quality_nodes = [n for n in tier_nodes if _far_quality(n)]
    soft_nodes = [n for n in tier_nodes if _far_quality_soft(n)]
    if not soft_nodes:
      return _commit_lost()

    reacq = float(self.sticky_live_reacquire_px)
    # Sticky rematch ao last-live mesmo longe (flicker breve do mesmo disco).
    if self._last_live_x is not None and self._last_live_y is not None:
      near_live = [n for n in soft_nodes if _jump_last_live(n) <= reacq]
      if near_live:
        return _commit_live(min(near_live, key=_score))

    max_jump = 42.0 + min(ref_dist * 0.12, 20.0)
    pool = quality_nodes if quality_nodes else soft_nodes
    near = [n for n in pool if _jump_lock(n) <= max_jump]
    if near:
      return _commit_live(min(near, key=_score))

    angled = [
      n
      for n in pool
      if abs(normalize_angle_deg(_node_angle(n) - prev_a)) <= 40.0
      and abs(n.distance_px - ref_dist) <= 28.0
    ]
    if angled:
      return _commit_live(min(angled, key=_score))

    # Expansão last-live: disco andou no minimapa mas ainda é o mesmo alvo.
    if self._last_live_x is not None and self._last_live_y is not None:
      rescue_r = max(reacq * 2.2, 52.0)
      expanded = [
        n
        for n in soft_nodes
        if _jump_last_live(n) <= rescue_r
        and abs(normalize_angle_deg(_node_angle(n) - prev_a)) <= 55.0
      ]
      if expanded:
        return _commit_live(min(expanded, key=_score))

    return _commit_lost()

  def _close_walk_blip_xy(
    self,
    scan: NodeScanResult,
    *,
    lock_x: float,
    lock_y: float,
  ) -> tuple[float, float] | None:
    """
    Centro do disco cinza *live* no minimapa (coords de tela do ROI).
    None se sumiu / colapsou na seta — não usar lock fantasma no pivô nem
    last-live congelado (coords de tela não rolam com o mapa).
    Durante close-walk: rematch curto e rejeita salto que afasta do pivô.
    """
    if self._lock is None:
      return None
    min_sep = float(self.close_walk_min_sep_px)
    # Close-walk: raio menor evita agarrar outro gray e congelar progresso.
    reacq = float(self.sticky_live_reacquire_px)
    if self._close_walking:
      reacq = min(reacq, 12.0)
    px, py = float(scan.player_x), float(scan.player_y)
    start_sep = float(self._close_walk_start_sep or 999.0)
    max_sep = start_sep + 6.0 if self._close_walking else 999.0
    live_pool = self._live_tier_nodes(scan, tier=self._lock.tier)
    near_live = [
      n
      for n in live_pool
      if n.distance_px >= min_sep
      and n.distance_px <= max_sep
      and (
        math.hypot(n.x - lock_x, n.y - lock_y) <= reacq
        or (
          self._last_live_x is not None
          and self._last_live_y is not None
          and math.hypot(n.x - self._last_live_x, n.y - self._last_live_y)
          <= reacq
        )
      )
    ]
    if not near_live:
      return None
    best = min(
      near_live,
      key=lambda n: (
        math.hypot(n.x - lock_x, n.y - lock_y),
        math.hypot(
          n.x - (self._last_live_x if self._last_live_x is not None else lock_x),
          n.y - (self._last_live_y if self._last_live_y is not None else lock_y),
        ),
        math.hypot(n.x - px, n.y - py),
      ),
    )
    return float(best.x), float(best.y)

  def _close_walk_rate(self) -> float:
    return max(float(self._close_walk_px_per_s), float(self.close_walk_min_rate_px_s))

  def _note_close_walk_progress(self, progress: float, now: float) -> None:
    """Atualiza progresso monotônico + taxa px/s + orçamento de tempo restante."""
    prev = float(self._close_walk_progress)
    gained = progress - prev
    if gained > 0.15:
      dt = 0.0
      if self._close_walk_last_progress_at is not None:
        dt = max(1e-3, now - self._close_walk_last_progress_at)
      if dt > 0.0:
        inst = gained / dt
        if self._close_walk_px_per_s <= 0.0:
          self._close_walk_px_per_s = inst
        else:
          self._close_walk_px_per_s = 0.65 * self._close_walk_px_per_s + 0.35 * inst
      self._close_walk_last_progress_at = now
      self._close_walk_progress = progress
      # Orçamento: tempo já andado + restante/taxa (+folga), teto = max_s.
      need = float(self.close_walk_px)
      remaining = max(0.0, need - progress)
      rate = self._close_walk_rate()
      started = self._close_walk_started_at or now
      elapsed = max(0.0, now - started)
      dynamic = elapsed + remaining / rate + 0.25
      cap = float(self.close_walk_max_s) if self.close_walk_max_s > 0 else dynamic
      self._close_walk_budget_s = min(cap, dynamic) if cap > 0 else dynamic
    else:
      self._close_walk_progress = max(prev, progress)

  def _begin_close_walk(
    self,
    scan: NodeScanResult,
    *,
    lock_x: float,
    lock_y: float,
  ) -> None:
    now = time.perf_counter()
    self._close_walking = True
    self._close_walk_done = False
    self._close_walk_started_at = now
    self._close_walk_progress = 0.0
    self._close_walk_blip_lost = False
    self._close_walk_last_progress_at = now
    self._close_walk_px_per_s = 0.0
    self._close_walk_lost_count = 0
    # Orçamento inicial: need/taxa_mín + folga, limitado por max_s.
    need = float(self.close_walk_px)
    rate = float(self.close_walk_min_rate_px_s)
    initial = need / max(rate, 1.0) + 0.35
    cap = float(self.close_walk_max_s) if self.close_walk_max_s > 0 else initial
    self._close_walk_budget_s = min(cap, initial) if cap > 0 else initial
    px, py = float(scan.player_x), float(scan.player_y)
    blip = self._close_walk_blip_xy(scan, lock_x=lock_x, lock_y=lock_y)
    if blip is not None:
      self._close_walk_start_sep = math.hypot(blip[0] - px, blip[1] - py)
    else:
      # Snapshot do lock/last-live só no start (baseline); updates usam live only.
      sx = (
        float(self._last_live_x)
        if self._last_live_x is not None
        else float(lock_x)
      )
      sy = (
        float(self._last_live_y)
        if self._last_live_y is not None
        else float(lock_y)
      )
      self._close_walk_start_sep = math.hypot(sx - px, sy - py)
      if self._close_walk_start_sep < self.close_walk_min_sep_px:
        self._close_walk_start_sep = float(self.arrive_px)
      self._close_walk_blip_lost = True
      self._close_walk_lost_count = 1

  def _update_close_walk_progress(
    self,
    scan: NodeScanResult,
    *,
    lock_x: float,
    lock_y: float,
  ) -> tuple[float, bool]:
    """
    progress = start_sep - cur_sep (quanto o blip se aproximou do pivô no minimapa).
    start_sep/cur_sep = hypot(blip_xy - player_pivot_xy) em pixels de tela do ROI.
    Retorna (progress, done). Nunca trata sep≈0 / dist zeroed como done.
    """
    need = float(self.close_walk_px)
    start_sep = float(self._close_walk_start_sep or self.arrive_px)
    px, py = float(scan.player_x), float(scan.player_y)
    blip = self._close_walk_blip_xy(scan, lock_x=lock_x, lock_y=lock_y)
    now = time.perf_counter()
    elapsed = 0.0
    if self._close_walk_started_at is not None:
      elapsed = now - self._close_walk_started_at

    if blip is not None:
      cur_sep = math.hypot(blip[0] - px, blip[1] - py)
      # Rejeita snap à seta: sep colapsada ≠ progresso instantâneo.
      if cur_sep >= self.close_walk_min_sep_px:
        self._close_walk_blip_lost = False
        self._close_walk_lost_count = 0
        self._note_close_walk_progress(max(0.0, start_sep - cur_sep), now)
      elif self._close_walk_progress >= need - 3.0 or self._close_walk_progress >= need * 0.8:
        # Quase no alvo e sumiu sob a seta — aceita como done.
        self._close_walk_progress = max(self._close_walk_progress, need)
        self._close_walk_blip_lost = False
        self._close_walk_lost_count = 0
      else:
        self._close_walk_blip_lost = True
        self._close_walk_lost_count += 1
    else:
      self._close_walk_blip_lost = True
      self._close_walk_lost_count += 1

    progress = float(self._close_walk_progress)
    if need <= 0 or progress >= need:
      return progress, True

    remaining = max(0.0, need - progress)
    rate = self._close_walk_rate()
    last_at = self._close_walk_last_progress_at or self._close_walk_started_at or now

    # Blip sumiu: estima o restante pela taxa (não segura W até max_s cheio).
    if self._close_walk_blip_lost and progress > 0.5:
      lost_for = max(0.0, now - last_at)
      est = progress + rate * lost_for
      if est >= need:
        self._close_walk_progress = need
        return need, True
      if (
        self._close_walk_lost_count >= self.close_walk_lost_frames
        and lost_for >= remaining / rate
      ):
        self._close_walk_progress = need
        return need, True

    # Estagnação: progresso congelado (rematch ruim / blip parado) → para.
    stall_s = float(self.close_walk_stall_s)
    if (
      stall_s > 0
      and (now - last_at) >= stall_s
      and progress > 0.5
    ):
      if progress >= need * 0.7 or remaining <= 5.0:
        self._close_walk_progress = need
        return need, True
      # Longe do alvo mas sem avanço: corta W (melhor undershoot que overshoot longo).
      return progress, True

    # Orçamento dinâmico + teto absoluto (sempre — não só com blip_lost).
    budget = self._close_walk_budget_s
    if budget is not None and budget > 0 and elapsed >= budget:
      return progress, True
    if self.close_walk_max_s > 0 and elapsed >= self.close_walk_max_s:
      return progress, True

    return progress, False

  def _resolve_bearing(
    self,
    *,
    legacy_arrow: Any,
    target: MiningNode,
    arrow: Any | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
  ) -> tuple[float | None, float | None]:
    tx = float(target.x if target_x is None else target_x)
    ty = float(target.y if target_y is None else target_y)
    if self.control_mode == "camera":
      if arrow is not None:
        raw = camera_align_bearing(
          arrow,
          legacy_arrow,
          tx,
          ty,
          prefer_facing=True,
          first_person=self.first_person,
        )
      else:
        raw = camera_heading_from_arrow(legacy_arrow, tx, ty)
      if raw is None:
        return self._smooth_heading, self._smooth_heading
      if self.camera_align_only:
        smoothed = raw
        self._smooth_heading = raw
      else:
        if self._smooth_heading is not None:
          jump = abs(normalize_angle_deg(raw - self._smooth_heading))
          if jump > self.camera_bearing_max_jump_deg:
            raw = self._smooth_heading
        alpha = max(self.heading_smooth_alpha, 0.55)
        smoothed = smooth_heading_error(self._smooth_heading, raw, alpha=alpha)
        self._smooth_heading = smoothed
      dot: float | None = None
      if arrow is not None:
        dot = target_ahead_dot_facing(
          arrow,
          tx,
          ty,
          legacy_arrow=legacy_arrow,
          first_person=self.first_person,
        )
      else:
        tip_x = legacy_arrow.arrow_tip_x
        tip_y = legacy_arrow.arrow_tip_y
        if tip_x is not None and tip_y is not None:
          px, py = legacy_arrow.pivot()
          _, dot = arrow_to_target_with_dot(px, py, tip_x, tip_y, tx, ty)
      return smoothed, dot

    aim = target
    if target_x is not None and target_y is not None:
      aim = replace(
        target,
        x=tx,
        y=ty,
        distance_px=math.hypot(
          tx - legacy_arrow.pivot()[0],
          ty - legacy_arrow.pivot()[1],
        ),
      )
    heading, self._smooth_heading = resolve_heading_error(
      arrow=legacy_arrow,
      target=aim,
      smooth_prev=self._smooth_heading,
      smooth_alpha=self.heading_smooth_alpha,
    )
    dot: float | None = None
    tip_x = legacy_arrow.arrow_tip_x
    tip_y = legacy_arrow.arrow_tip_y
    if tip_x is not None and tip_y is not None:
      px, py = legacy_arrow.pivot()
      _, dot = arrow_to_target_with_dot(px, py, tip_x, tip_y, tx, ty)
    return heading, dot

  def update_facing(self, raw_facing: float | None) -> float | None:
    self._smooth_facing = stable_facing_deg(
      raw_facing,
      self._smooth_facing,
      max_jump_deg=self.facing_max_jump_deg,
    )
    return self._smooth_facing

  def evaluate(
    self,
    scan: NodeScanResult | None,
    *,
    legacy_arrow: Any,
    facing_deg: float | None,
    arrow: Any | None = None,
  ) -> PursuitOutput:
    del facing_deg
    if scan is None or self._lock is None:
      return PursuitOutput(
        target=None,
        bearing_deg=None,
        target_dot=None,
        dist_px=0.0,
        aligned=False,
        arrived=False,
        nav_status="sem-alvo",
        display_lock=None,
      )

    arrive_use = float(self.arrive_px)

    target = self.resolve_target(scan)
    keep_lock = (
      self._sticky_id
      or self._hold_for_mine
      or self._fine_aligning
      or self._fine_aligned
      or self._close_walking
      or self._final_approaching
    )
    if target is None:
      if self._lock is not None and keep_lock:
        px, py = scan.player_x, scan.player_y
        target = self.detector.node_from_lock(
          self._lock, player_x=px, player_y=py
        )
      else:
        return PursuitOutput(
          target=None,
          bearing_deg=None,
          target_dot=None,
          dist_px=0.0,
          aligned=False,
          arrived=False,
          nav_status="sem-alvo",
          display_lock=None,
        )

    is_live = not bool(getattr(target, "ghost", False))
    # Centro live: se houver blip matched no raio sticky, snap lock + bearing nele.
    # Evita linha verde / rumo no XY congelado enquanto o disco cinza andou.
    lock_x = float(self._lock.locked_x if self._lock is not None else target.x)
    lock_y = float(self._lock.locked_y if self._lock is not None else target.y)
    if self._lock is not None:
      reacq = float(self.sticky_live_reacquire_px)
      # Close-walk: rematch curto + não aceitar blip mais longe que o start_sep.
      if self._close_walking:
        reacq = min(reacq, 12.0)
      live_pool = self._live_tier_nodes(scan, tier=self._lock.tier)
      max_sep = 999.0
      if self._close_walking and self._close_walk_start_sep is not None:
        max_sep = float(self._close_walk_start_sep) + 6.0
      near_live = [
        n
        for n in live_pool
        if n.distance_px <= max_sep
        and (
          math.hypot(n.x - lock_x, n.y - lock_y) <= reacq
          or (
            self._last_live_x is not None
            and self._last_live_y is not None
            and math.hypot(n.x - self._last_live_x, n.y - self._last_live_y) <= reacq
          )
        )
      ]
      if near_live:
        best = min(
          near_live,
          key=lambda n: (
            math.hypot(n.x - lock_x, n.y - lock_y),
            math.hypot(
              n.x - (self._last_live_x if self._last_live_x is not None else lock_x),
              n.y - (self._last_live_y if self._last_live_y is not None else lock_y),
            ),
          ),
        )
        self._lock.locked_x = float(best.x)
        self._lock.locked_y = float(best.y)
        self._lock.lost_frames = 0
        self._lock.last_distance_px = float(best.distance_px)
        self._lock.last_bearing_deg = bearing_deg(
          scan.player_x, scan.player_y, best.x, best.y
        )
        self._last_live_x = float(best.x)
        self._last_live_y = float(best.y)
        lock_x, lock_y = float(best.x), float(best.y)
        if getattr(target, "ghost", False):
          target = best
        is_live = True
    dist = float(math.hypot(lock_x - scan.player_x, lock_y - scan.player_y))
    bearing, dot = self._resolve_bearing(
      legacy_arrow=legacy_arrow,
      target=target,
      arrow=arrow,
      target_x=lock_x,
      target_y=lock_y,
    )
    lost_frames = int(self._lock.lost_frames) if self._lock else 0

    # Progresso de approach: só distância live.
    if is_live:
      if self._approach_min_dist is None or dist < self._approach_min_dist - 0.3:
        self._approach_min_dist = dist

    started_far = float(self._lock.pick_distance_px) >= arrive_use + 12.0
    # ≤arrive: para W e mantém lock (live ou sticky ghost) — nunca limpa.
    arrived = started_far and dist <= arrive_use
    if self.control_mode == "camera" and self.camera_align_only:
      arrived = False

    vnav = self.visual_nav.update(
      pivot_x=legacy_arrow.pivot()[0],
      pivot_y=legacy_arrow.pivot()[1],
      tip_x=legacy_arrow.arrow_tip_x,
      tip_y=legacy_arrow.arrow_tip_y,
      target_x=lock_x,
      target_y=lock_y,
      tile_dist_px=dist,
      tile_arrive_px=self.arrive_px,
      heading_error_deg=bearing,
    )
    aligned = bearing is not None and abs(bearing) <= self.visual_nav.align_deg

    # Overlay lock = centro do nó live (mesmo endpoint da linha verde).
    show_x, show_y = lock_x, lock_y
    # Sticky: nunca brg=NA — continua navegando ate arrive.
    if bearing is None:
      bearing = self._smooth_heading if self._smooth_heading is not None else 0.0

    display = V2TargetLock(
      track_id=int(self._lock.node_id or self._display_id),
      x=show_x,
      y=show_y,
      tier=str(target.tier).lower(),
      lost_frames=lost_frames,
      pinned=(
        self._sticky_id
        or self._fine_aligning
        or self._fine_aligned
        or self._close_walking
        or self._close_walk_done
        or self._final_approaching
      ),
    )

    # Chegou perto: para W, mantém lock, fine-align da câmera (sem limpar).
    if (arrived or self._fine_aligning) and not self._fine_aligned:
      self._fine_aligning = True
      self._sticky_id = True
      abs_brg = abs(bearing) if bearing is not None else 999.0
      if abs_brg <= self.fine_align_deadband_deg:
        self._fine_align_ok_frames += 1
      else:
        self._fine_align_ok_frames = 0
      if self._fine_align_ok_frames >= self.fine_align_settle_frames:
        self._fine_aligned = True
        self._fine_aligning = False
        self._begin_close_walk(scan, lock_x=lock_x, lock_y=lock_y)
        # Cai no bloco close_walk abaixo no mesmo tick.
      else:
        return PursuitOutput(
          target=target,
          bearing_deg=bearing,
          target_dot=dot,
          dist_px=dist,
          aligned=abs_brg <= self.fine_align_deadband_deg,
          arrived=True,
          nav_status="FINE-ALIGN",
          display_lock=display,
          move_phase="fine_align",
        )

    # Fine-align ok → W até close_walk_px de progresso blip→pivot (sem câmera).
    # NÃO completar via dist/target.distance_px (HUD zera perto do nó).
    if self._close_walking and not self._close_walk_done:
      progress, done = self._update_close_walk_progress(
        scan, lock_x=lock_x, lock_y=lock_y
      )
      if done:
        self._close_walking = False
        self._close_walk_done = True
        # Pós close-walk — brain entra em FINAL probe E (sem W loop).
        return PursuitOutput(
          target=target,
          bearing_deg=bearing,
          target_dot=dot,
          dist_px=dist,
          aligned=True,
          arrived=True,
          nav_status="CLOSE-WALK done",
          display_lock=display,
          move_phase="close_done",
        )
      need = float(self.close_walk_px)
      lost_tag = " lost" if self._close_walk_blip_lost else ""
      return PursuitOutput(
        target=target,
        bearing_deg=bearing,
        target_dot=dot,
        dist_px=dist,
        aligned=True,
        arrived=True,
        nav_status=f"CLOSE-WALK {progress:.1f}/{need:g}px{lost_tag}",
        display_lock=display,
        move_phase="close_walk",
      )

    # Pós close-walk: close_done até brain chamar begin_final_approach (probe E).
    if self._close_walk_done and not self._final_approaching:
      return PursuitOutput(
        target=target,
        bearing_deg=bearing,
        target_dot=dot,
        dist_px=dist,
        aligned=True,
        arrived=True,
        nav_status="CLOSE-WALK done",
        display_lock=display,
        move_phase="close_done",
      )

    if self._final_approaching:
      return PursuitOutput(
        target=target,
        bearing_deg=bearing,
        target_dot=dot,
        dist_px=dist,
        aligned=True,
        arrived=True,
        nav_status="FINAL-APPROACH",
        display_lock=display,
        move_phase="final_approach",
      )

    if self._fine_aligned:
      return PursuitOutput(
        target=target,
        bearing_deg=bearing,
        target_dot=dot,
        dist_px=dist,
        aligned=True,
        arrived=True,
        nav_status="ALINHADO",
        display_lock=display,
        move_phase="aligned",
      )

    return PursuitOutput(
      target=target,
      bearing_deg=bearing,
      target_dot=dot,
      dist_px=dist,
      aligned=aligned,
      arrived=False,
      nav_status=self.visual_nav.overlay_metrics(vnav),
      display_lock=display,
      move_phase=self.turn.phase,
    )

  def begin_final_approach(self) -> None:
    """Ativa FINAL (probe E pós close-walk / abort-retry)."""
    self._final_approaching = True
    self._close_walking = False
    self._close_walk_done = True
    self._fine_aligned = True

  def note_progress(
    self,
    dist_px: float,
    *,
    bearing_deg: float | None = None,
  ) -> bool:
    """Retorna True se estagnado tempo suficiente para pular nó."""
    now = time.perf_counter()

    if self.turn.is_turning() and bearing_deg is not None:
      brg_gain = self.turn.bearing_progress(bearing_deg)
      if brg_gain is not None and brg_gain > 1.5:
        self._stagnant_since = None
        return False

    if bearing_deg is not None and abs(bearing_deg) > self.turn.align_deg:
      if self._prev_abs_bearing is not None:
        if self._prev_abs_bearing - abs(bearing_deg) > 1.0:
          self._stagnant_since = None
      self._prev_abs_bearing = abs(bearing_deg)
      return False

    if (
      self._approach_min_dist is None
      or dist_px < self._approach_min_dist - 0.8
    ):
      self._approach_min_dist = dist_px
      self._stagnant_since = None
      if bearing_deg is not None:
        self._prev_abs_bearing = abs(bearing_deg)
      return False
    if self._stagnant_since is None:
      self._stagnant_since = now
      return False
    if (
      now - self._stagnant_since >= self.stagnation_s
      and dist_px > self.arrive_px * 2.0
    ):
      self._stagnant_since = None
      return True
    return False

  def _post_stuck_realign_done(
    self,
    bearing_deg: float | None,
    *,
    target_dot: float | None,
  ) -> bool:
    """True quando rumo ao lock está dentro de stuck_align_deg (pode liberar W)."""
    if bearing_deg is None:
      return False
    if target_dot is not None and target_dot < 0:
      return False
    return abs(float(bearing_deg)) <= float(self.stuck_align_deg)

  def walk(
    self,
    bearing_deg: float | None,
    *,
    dist_px: float = 0.0,
    target_dot: float | None = None,
    fine_align: bool = False,
    close_walk: bool = False,
  ) -> str:
    if self.control_mode == "camera":
      force_realign = bool(self._need_post_stuck_realign) and not fine_align and not close_walk
      action = self.walker.tick(
        walk=not fine_align,
        bearing_deg=bearing_deg,
        dist_px=dist_px,
        target_dot=target_dot,
        fine_align=fine_align,
        close_walk=close_walk,
        force_realign=force_realign,
      )
      # Só libera W depois de pelo menos 1 tick de realign pós-D e brg ok.
      if force_realign and self._post_stuck_realign_done(
        bearing_deg, target_dot=target_dot
      ):
        self._need_post_stuck_realign = False
      return action

    if fine_align:
      return self.walker.stop()
    if close_walk:
      return self.walker.update(0.0, walk=True)

    # Facing mode: só limpa o flag; camera trata force_realign acima.
    if self._need_post_stuck_realign and self._post_stuck_realign_done(
      bearing_deg, target_dot=target_dot
    ):
      self._need_post_stuck_realign = False

    phase = self.turn.update(bearing_deg)
    steer = self.turn.steering_bearing(bearing_deg)
    if phase == "walk":
      return self.walker.update(steer, walk=True)
    if phase.startswith("align-"):
      return self.walker.update(steer, walk=True)
    if phase == "settle":
      return self.walker.stop()
    return self.walker.stop()

  def stop_walk(self) -> str:
    return self.walker.stop()

  def interact(self) -> str:
    return self.walker.interact()
