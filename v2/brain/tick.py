"""
Máquina de estados v2 — abordagem + fine-align + close-walk + probe E.

Ao chegar (dist ≤ arrive_px): para W, mantém o lock, ajusta a câmera com
pulsos finos até a ponta amarela no centro do nó (|brg| ≤ deadband), depois
segura W até close_walk_px de progresso no minimapa (blip→pivot).

Após close-walk: FINAL_APPROACH —
  1) standstill (final_wait_before_e_ms) → probe E inicial
     (final_probe_e_ms, tipicamente 750ms);
  2) se Mining ore present mid-hold → READY mine-hold (keep E até label gone
     → SCAN restart);
  3) se miss → até final_pulse_max ciclos: pulse W (final_pulse_w_ms) →
     wait (final_wait_after_w_ms) → standstill → probe E de novo;
     present mid-hold → READY;
  4) após final_pulse_max ciclos W+E sem label → mark_done (soft-avoid) +
     SCAN + auto-lock (outro nó; mesmo ciclo que MINING done). F8 (=proximo)
     ainda avança via `_advance_after_node`.

FINAL_APPROACH (pós close-walk):
  Sub-passos: e (standstill) → holding_e → (miss) w → wait_w → e → …
  (≤ final_pulse_max W). Antes de cada probe E: parado + final_wait_before_e_ms.
  READY entry — só label real durante hold E (mesmo detector; sem atalho
  STILL_MINING ambient):
    - `_ore_label_present`: found|raw_hit|score≥hold_min
    - Só em `_fa_sub == holding_e` (ou e_held_for_mine já True)
    - Antes do 1º probe / fora de holding_e: não READY
    - Sem W enquanto label present / mining
    - engage_threshold só para log “quase” abaixo de hold_min
  MINING done (READY exit): label **ausente** (`_ore_frame_absent`:
    gone_threshold / mediana baixa / soft drop) por gone_confirm_frames, ou
    mine_hold_timeout_s / below-hold — oposto de present.
  Em READY: W/A/S/D sempre soltos; poll Mining ore; quando a label some por
    gone_confirm_frames → solta E, log MINING done → SCAN + auto-lock
    (ciclo completo de novo). Probe miss (pulse-max) → mark_done + SCAN
    + auto-lock (outro alvo).
  Fim de mine: keep E se present | mediana≥hold_min.
  Absent só se (mediana/score) < gone_threshold, ou (abaixo de hold_min
  E queda ≥ drop do peak). Drop NÃO aplica com score ainda alto
  (shake 0.95→0.84 com barra a 50% NÃO solta E).
  FP pós-gone platô ≥hold_min → mine_hold_timeout_s (não ore-gone).
  Timeout: mine_hold_timeout_s (default 20s) força MINING done
  independentemente do score; mine_hold_below_match_s + streak
  < hold_min também força fim.
  Pulse-max miss / timeout FINAL → mark_done + SCAN + auto-lock (não
  COOLDOWN idle).
  SCAN/GOTO STILL_MINING → READY só se e_held_for_mine (mine real) E label
  present — nunca ore score ambient (match/hold) durante fine-align / GOTO /
  antes de close-walk + probe E. Após PAROU: fine-align → close-walk → probe.
  Nunca pulse-W / walk / stuck-D / auto-lock com present no FINAL,
  e_held_for_mine, READY mine-hold.

Auto lock_nearest no armamento (F6 → SCAN), após MINING done e após
  pulse-max / probe miss (outro nó). Abort longe legado → SCAN.
Nó já marcado done em READY / probe-miss (`mark_done` / `_done_xy`) —
  não re-trava o mesmo.
SCAN sem nós (lock_nearest=None): sub-estado SCAN_SPIN — pulsos de câmera
  à esquerda (~360° via pixels_per_deg); cada pulse tenta lock de novo;
  lock cancela o spin → GOTO; após 360° pausa breve e repete (até F6 off).
GOTO walk sem progresso de dist (`stuck_idle_s`) → até `stuck_d_max_attempts`
  pulses D (`stuck_d_hold_ms`); após cada D, realign de câmera (sem W) até
  |brg| ≤ stuck_align_deg / walk_max; só então retoma W. Depois mark_stuck + SCAN.
  Cada give-up append na blacklist (multi-nó); limpa em READY (Mining ore).
  stuck_idle só em GOTO walk — nunca em READY (phase branch separado).
  `_stuck_d_attempts` reseta só em novo lock / mark_done / reset (não em
  jitter de dist — senão nunca chega a 3× D no mesmo alvo).
Lost-target: só quando `pursuit.target is None` (ghost abandon). Frame sem
  scan/arrow NÃO demote a SCAN (evita lock órfão + linha verde congelada + spin).
Após lost-target: cooldown `lock_reacquire_cooldown_s`; spin-gate só segura
  se não houver disco live travável (fora do ghost-avoid).
F8 = forçar próximo alvo (lock_next).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from logger import mlog
from node_detector import NodeScanResult
from v2.core.types import FrameContext, Phase, TargetLock as V2TargetLock
from v2.navigation.bearing import FIRST_PERSON_FORWARD_DEG, forward_heading_deg
from v2.navigation.pursuit_controller import PursuitController
from v2.perception.mining_ore_detector import MiningOreDetector


class Brain:
  def __init__(self, cfg: dict[str, Any], node_detector: Any) -> None:
    self.pursuit = PursuitController(cfg, node_detector)
    self.mining_ore = MiningOreDetector(cfg)
    self.phase = Phase.SCAN
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    # True após F6 enable / MINING done / abort final; False após READY.
    self._allow_auto_lock = False
    self._final_started_at: float | None = None
    self._pulse_count = 0
    # Sub-passos: e (standstill) → holding_e → w → wait_w → e → …
    self._fa_sub = "e"
    self._fa_sub_at = 0.0
    # READY: hit forte visto → conta frames absent até fim (gone sustain).
    self._mining_label_seen = False
    self._mining_had_strong = False
    self._ore_gone_streak = 0
    self._ore_below_match_streak = 0
    self._ore_peak = 0.0
    self._mining_ready_at: float | None = None
    nav = cfg.get("navigation", {})
    self.final_pulse_max = max(1, int(nav.get("final_pulse_max", 5)))
    self.final_pulse_w_ms = float(nav.get("final_pulse_w_ms", 350.0))
    # 0 = desliga abort por tempo (só pulse_max reinicia).
    self.final_approach_timeout_s = float(nav.get("final_approach_timeout_s", 0.0))
    self.final_wait_after_w_ms = float(nav.get("final_wait_after_w_ms", 300.0))
    self.final_wait_after_e_ms = float(nav.get("final_wait_after_e_ms", 280.0))
    # Parado antes de cada keydown E (inicial pós close-walk e após wait_w).
    self.final_wait_before_e_ms = float(nav.get("final_wait_before_e_ms", 200.0))
    self.final_probe_e_ms = float(nav.get("final_probe_e_ms", 750.0))
    # Abort FINAL ainda perto do lock → retry mesmo alvo (não pular nó longe).
    self.final_abort_retry_px = float(
      nav.get(
        "final_abort_retry_px",
        nav.get("done_radius_px", nav.get("sticky_enter_px", 40.0)),
      )
    )
    ore_cfg = nav.get("mining_ore", {})
    # Present (READY entry + mine-hold keep): single floor hold_min /
    # present_threshold alias — mesmo detector/ROI start+end.
    if "hold_min" in ore_cfg:
      self.ore_hold_min = float(ore_cfg["hold_min"])
    elif "present_threshold" in ore_cfg:
      self.ore_hold_min = float(ore_cfg["present_threshold"])
    else:
      self.ore_hold_min = 0.70
    self.ore_present_threshold = float(self.ore_hold_min)
    # Soft floor só p/ log “quase” abaixo de hold_min.
    self.ore_engage_threshold = float(
      ore_cfg.get("engage_threshold", self.ore_hold_min)
    )
    # Legacy config (ignorado p/ READY — present usa hold_min).
    self.ore_ready_confirm_frames = max(
      1, int(ore_cfg.get("ready_confirm_frames", 3))
    )
    self._ore_engage_streak = 0
    # Alias legado → hold_min (present unificado).
    self.ore_final_keep_min = float(
      ore_cfg.get("final_keep_min", self.ore_hold_min)
    )
    # Absent floor: abaixo do engage típico; noise / label sumiu.
    self.ore_gone_threshold = float(ore_cfg.get("gone_threshold", 0.40))
    # Soft drop só abaixo de hold_min (nunca com score ainda alto).
    self.ore_gone_drop_from_peak = float(
      ore_cfg.get("gone_drop_from_peak", 0.20)
    )
    self.ore_gone_confirm_frames = max(
      1, int(ore_cfg.get("gone_confirm_frames", 8))
    )
    self.ore_score_smooth_frames = max(
      1, int(ore_cfg.get("score_smooth_frames", 5))
    )
    self._ore_score_hist: deque[float] = deque(
      maxlen=self.ore_score_smooth_frames
    )
    # Hard timeout from READY mine-hold start — backup if gone never fires.
    self.ore_mine_hold_timeout_s = float(
      ore_cfg.get("mine_hold_timeout_s", 20.0)
    )
    # Após N s com score < hold_min por M frames → force end (banda cinza).
    self.ore_mine_hold_below_match_s = float(
      ore_cfg.get("mine_hold_below_match_s", 25.0)
    )
    # Após MINING done: TTL legado (ore ambient já não resume READY).
    self._post_mine_ore_gate_s = float(
      ore_cfg.get("post_mine_ore_gate_s", 2.0)
    )
    self._post_mine_ore_gate_until = 0.0
    cam = nav.get("camera", {})
    self.scan_spin_enabled = bool(cam.get("scan_spin_enabled", True))
    self.scan_spin_total_deg = max(1.0, float(cam.get("scan_spin_total_deg", 360.0)))
    self.scan_spin_pulse_deg = max(0.5, float(cam.get("scan_spin_pulse_deg", 15.0)))
    self.scan_spin_interval_ms = float(
      cam.get("scan_spin_interval_ms", cam.get("look_interval_ms", 130.0))
    )
    # Pausa entre revoluções completas (0 = sem pausa).
    self.scan_spin_pause_ms = max(0.0, float(cam.get("scan_spin_pause_ms", 400.0)))
    # Sub-estado SCAN: spin left quando não há nós.
    self._scan_spinning = False
    self._scan_spin_yaw = 0.0
    self._scan_spin_pause_until = 0.0
    # Após ghost abandon / lost-target: pausa breve antes de re-lock (anti-thrash).
    self._lock_cooldown_until = 0.0
    self.lock_reacquire_cooldown_s = float(
      nav.get("lock_reacquire_cooldown_s", 0.4)
    )
    # Após lost-target: prefere 1 revolução SCAN_SPIN antes de re-travar fagulha.
    # Disco live travável (fora do ghost-avoid) libera o gate sem esperar 360°.
    self._post_lost_spin_gate = False

  @property
  def walker(self):
    return self.pursuit.walker

  def _lock_facing(self, ctx: FrameContext | None) -> float | None:
    if ctx is None:
      return FIRST_PERSON_FORWARD_DEG if self.pursuit.first_person else None
    legacy = ctx.meta.get("legacy_arrow")
    return forward_heading_deg(
      ctx.arrow,
      legacy,
      first_person=bool(ctx.meta.get("first_person", self.pursuit.first_person)),
    )

  def reset_session(self) -> None:
    self.phase = Phase.SCAN
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._allow_auto_lock = True
    self._lock_cooldown_until = 0.0
    self._post_lost_spin_gate = False
    self._reset_final_state()
    self._reset_mining_end_state()
    self._reset_scan_spin()
    self._post_mine_ore_gate_until = 0.0
    clear = getattr(self.walker, "clear_e_hold", None)
    if callable(clear):
      clear()
    self.pursuit.reset()
    self.mining_ore.reset()

  def _reset_scan_spin(self) -> None:
    self._scan_spinning = False
    self._scan_spin_yaw = 0.0
    self._scan_spin_pause_until = 0.0

  def _arm_lock_cooldown(self) -> None:
    cool = float(self.lock_reacquire_cooldown_s)
    if cool > 0:
      self._lock_cooldown_until = time.perf_counter() + cool

  def _arm_lost_target_recovery(self) -> None:
    """Clear → cooldown → spin once antes do próximo auto-lock."""
    self._arm_lock_cooldown()
    self._post_lost_spin_gate = True
    self._reset_scan_spin()

  def _clear_post_lost_spin_gate(self) -> None:
    if self._post_lost_spin_gate:
      self._post_lost_spin_gate = False
      mlog("[v2] SCAN — post-lost spin done, auto-lock armed")

  def _reset_final_state(self) -> None:
    self._final_started_at = None
    self._pulse_count = 0
    self._fa_sub = "e"
    self._fa_sub_at = 0.0
    self._ore_engage_streak = 0
    self.mining_ore.reset()

  def _reset_mining_end_state(self) -> None:
    self._mining_label_seen = False
    self._mining_had_strong = False
    self._ore_gone_streak = 0
    self._ore_below_match_streak = 0
    self._ore_peak = 0.0
    self._ore_score_hist.clear()
    self._mining_ready_at = None

  def _ore_score_median(self) -> float | None:
    """Mediana dos últimos score_smooth_frames (absorve shake da câmera)."""
    if not self._ore_score_hist:
      return None
    vals = sorted(self._ore_score_hist)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
      return float(vals[mid])
    return (float(vals[mid - 1]) + float(vals[mid])) / 2.0

  def _ore_frame_absent(
    self,
    *,
    score: float,
    thr: float,
    raw_hit: bool,
    found: bool,
  ) -> bool:
    """
    Label ausente neste frame?

    Presente (keep E) se: found | raw_hit | score ≥ hold_min |
    mediana recente ≥ hold_min.
    Ausente se (mediana/score) < gone_threshold, OU já abaixo de
    hold_min com queda ≥ drop do peak (soft). Drop NÃO aplica enquanto
    score/mediana ≥ hold_min — shake 0.95→0.84 com barra a 50% hold.
    thr (match) unused here — hold usa hold_min; caller passa thr p/ API.
    """
    _ = thr
    self._ore_score_hist.append(float(score))

    if found or raw_hit:
      if score > self._ore_peak:
        self._ore_peak = score
      return False

    hold = self.ore_hold_min
    if score >= hold and score > self._ore_peak:
      self._ore_peak = score

    med = self._ore_score_median()
    if score >= hold or (med is not None and med >= hold):
      return False

    # Já abaixo de hold_min: decide absent via gone floor ou soft drop.
    hist_n = len(self._ore_score_hist)
    use_med = med is not None and hist_n >= min(3, self.ore_score_smooth_frames)
    eff = float(med) if use_med else float(score)

    if eff < self.ore_gone_threshold:
      return True

    peak = self._ore_peak
    drop = self.ore_gone_drop_from_peak
    if drop > 0 and peak > 0 and score < (peak - drop):
      return True
    return False

  def _mining_busy(
    self,
    ctx: FrameContext,
    *,
    respect_post_mine_gate: bool = False,
  ) -> tuple[bool, Any]:
    """
    True se ainda mining ativo: e_held_for_mine de um mine real.

    Ore score ambient (match/hold/found) NÃO conta — evita READY via
    STILL_MINING em fine-align / GOTO / SCAN antes de close-walk + probe E.

    respect_post_mine_gate: legado (noop sem path ore-strong); E held
    sempre busy.
    """
    del respect_post_mine_gate  # legado API; ore ambient não resume
    hit = self._detect_mining_ore(ctx)
    e_held = getattr(self.walker, "e_held_for_mine", False) is True
    if e_held:
      return True, hit
    return False, hit

  def _can_resume_still_mining(self, hit: Any) -> bool:
    """
    STILL_MINING → READY só com E já held (mine real) E label present.
    Score ambient sozinho (ex. ore=0.70 no fine-align) → False.
    """
    if getattr(self.walker, "e_held_for_mine", False) is not True:
      return False
    return self._ore_label_present(hit)

  def _resume_ready_still_mining(self, hit: Any) -> tuple[str, str]:
    """
    Resume READY após mine real interrompido (E ainda held + label).
    Hard-gate: sem e_held ou sem label → não entra READY.
    """
    score = float(getattr(hit, "score", 0.0) or 0.0)
    if not self._can_resume_still_mining(hit):
      self.pursuit.stop_walk()
      return "goto-hold", f"GOTO hold (no still-mining ore={score:.3f})"
    if self.pursuit.v1_lock is not None:
      self.pursuit.reset()
    else:
      self.pursuit.stop_walk()
      reset_d = getattr(self.pursuit, "_reset_stuck_d_attempts", None)
      if callable(reset_d):
        reset_d()
      reset_idle = getattr(self.pursuit, "_reset_stuck_idle", None)
      if callable(reset_idle):
        reset_idle()
    self._reset_scan_spin()
    self._post_mine_ore_gate_until = 0.0
    return (
      self._enter_ready(reason="STILL_MINING", score=score),
      f"READY still-mining ore={score:.3f}",
    )

  def _scan(self, ctx: FrameContext) -> NodeScanResult | None:
    scan = ctx.meta.get("scan")
    return scan if isinstance(scan, NodeScanResult) else None

  def _try_auto_lock(
    self, scan: NodeScanResult | None, ctx: FrameContext | None = None
  ) -> bool:
    """lock_nearest inicial (F6 arm). Retorna True se passou a GOTO."""
    if not self._allow_auto_lock:
      return False
    # Ainda mining (E / ore forte): não trava outro nó.
    if ctx is not None:
      busy, _hit = self._mining_busy(ctx, respect_post_mine_gate=True)
      if busy:
        return False
    # SCAN nunca deve carregar lock órfão (linha verde congelada + spin sem alvo).
    if self.pursuit.v1_lock is not None:
      self.pursuit.reset()
    if scan is None:
      return False
    if time.perf_counter() < self._lock_cooldown_until:
      return False
    pick = self.pursuit.lock_nearest(scan, facing_deg=self._lock_facing(ctx))
    if pick is None:
      return False
    # Gate pós-lost: sem candidato = spin; com disco live = trava já.
    if self._post_lost_spin_gate:
      mlog(
        f"[v2] SCAN — live node during post-lost gate, lock "
        f"dist={pick.distance_px:.0f}px"
      )
    self.phase = Phase.GOTO
    # Lock ativo: não re-armar auto-lock até lost / mining done / abort.
    self._allow_auto_lock = False
    self._lock_cooldown_until = 0.0
    self._post_lost_spin_gate = False
    mlog(
      f"[v2] SCAN -> lock dist={pick.distance_px:.0f}px tier={pick.tier}"
    )
    return True

  def _tick_scan_or_spin(
    self,
    scan: NodeScanResult | None,
    ctx: FrameContext,
    *,
    active: bool,
  ) -> tuple[str, str]:
    """
    SCAN com auto-lock: tenta lock_nearest; se vazio e spin ligado,
    pulsos de câmera à esquerda até ~scan_spin_total_deg, depois pausa
    e repete. Lock a qualquer momento cancela o spin → GOTO.
    """
    # Mine real (E held) + label: volta READY. Ore ambient sozinho → não.
    busy, hit = self._mining_busy(ctx, respect_post_mine_gate=True)
    if busy and self._can_resume_still_mining(hit):
      return self._resume_ready_still_mining(hit)
    if busy:
      self.pursuit.stop_walk()
      return "scan-hold", "SCAN hold (e held, label absent)"

    was_spinning = self._scan_spinning
    if self._try_auto_lock(scan, ctx):
      if was_spinning:
        mlog("[v2] SCAN — lock during spin, cancel")
      self._reset_scan_spin()
      return "scan-goto", ""

    self.pursuit.stop_walk()

    if not self.scan_spin_enabled or not active:
      self._reset_scan_spin()
      # Sem spin: libera gate — só o cooldown impede re-lock.
      self._post_lost_spin_gate = False
      return "scan", "SCAN"

    now = time.perf_counter()
    if not self._scan_spinning:
      self._scan_spinning = True
      self._scan_spin_yaw = 0.0
      self._scan_spin_pause_until = 0.0
      mlog("[v2] SCAN — no nodes, spinning left 360")

    total = self.scan_spin_total_deg
    nav = f"SCAN_SPIN left {self._scan_spin_yaw:.0f}/{total:.0f}"

    if now < self._scan_spin_pause_until:
      return "scan-spin-pause", f"{nav} pause"

    look = getattr(self.walker, "look_yaw_deg", None)
    if not callable(look):
      return "scan", nav

    # Esquerda = yaw negativo (mesmo sinal de bearing; look_invert no walker).
    action = look(
      -self.scan_spin_pulse_deg,
      interval_ms=self.scan_spin_interval_ms,
    )
    if str(action).startswith("spin-look"):
      self._scan_spin_yaw += self.scan_spin_pulse_deg
      if self._scan_spin_yaw >= total - 1e-6:
        self._scan_spin_yaw = 0.0
        if self.scan_spin_pause_ms > 0:
          self._scan_spin_pause_until = now + (
            self.scan_spin_pause_ms / 1000.0
          )
        # Uma revolução completa libera o gate pós lost-target.
        self._clear_post_lost_spin_gate()
        mlog("[v2] SCAN — no nodes, spinning left 360")
      nav = f"SCAN_SPIN left {self._scan_spin_yaw:.0f}/{total:.0f}"
    elif action == "spin-wait":
      return "scan-spin-wait", nav
    return str(action), nav

  def _advance_after_node(self, ctx: FrameContext, *, reason: str) -> None:
    """Só via F8 — nunca automático pós READY/COOLDOWN."""
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._allow_auto_lock = False
    clear = getattr(self.walker, "clear_e_hold", None)
    if callable(clear):
      clear()
    self._reset_final_state()
    self._reset_mining_end_state()
    scan = self._scan(ctx)
    if scan is None:
      self.phase = Phase.SCAN
      mlog(f"[v2] {reason} — sem scan.")
      return
    pick = self.pursuit.lock_next(scan, facing_deg=self._lock_facing(ctx))
    if pick is None:
      self.phase = Phase.SCAN
      mlog(f"[v2] {reason} — sem proximo blip.")
    else:
      self.phase = Phase.GOTO
      mlog(
        f"[v2] {reason} → lock dist={pick.distance_px:.0f}px "
        f"tier={pick.tier}"
      )

  def try_preview_lock(self, ctx: FrameContext) -> None:
    if self.pursuit.v1_lock is not None:
      return
    scan = self._scan(ctx)
    if scan is None:
      return
    self.pursuit.lock_nearest(scan, facing_deg=self._lock_facing(ctx))

  def _ore_strong(self, hit: Any) -> bool:
    """Hit forte: found / raw_hit / score ≥ match_threshold (detector)."""
    score = float(getattr(hit, "score", 0.0) or 0.0)
    thr = float(self.mining_ore.threshold)
    return (
      bool(getattr(hit, "found", False))
      or bool(getattr(hit, "raw_hit", False))
      or score >= thr
    )

  def _ore_label_present(self, hit: Any) -> bool:
    """
    Label Mining ore presente — mesmo floor do mine-hold (keep E).

    found | raw_hit | score ≥ hold_min (ou present_threshold).
    READY entry no FINAL (após probe) e stop-W usam isto; mine-hold
    absent ainda acrescenta mediana / gone_threshold (oposto).
    """
    score = float(getattr(hit, "score", 0.0) or 0.0)
    if bool(getattr(hit, "found", False)) or bool(getattr(hit, "raw_hit", False)):
      return True
    hold = float(self.ore_present_threshold)
    return score >= hold

  def _ore_engage(self, hit: Any) -> bool:
    """
    Soft floor só em FINAL — log “quase” abaixo de hold_min.
    Presente (≥hold_min) também conta; fora de FINAL: False.
    """
    if self.phase != Phase.FINAL_APPROACH:
      return False
    if self._ore_label_present(hit):
      return True
    score = float(getattr(hit, "score", 0.0) or 0.0)
    floor = min(float(self.ore_engage_threshold), float(self.ore_final_keep_min))
    return score >= floor

  def _ore_final_keep(self, hit: Any) -> bool:
    """Bloqueia W / keep — present após probe (mesmo gate que READY)."""
    if self.phase != Phase.FINAL_APPROACH:
      return False
    if not self._final_probe_started():
      return False
    return self._ore_label_present(hit)

  def _final_probe_started(self) -> bool:
    """True após begin_probe_e / já mining (ciclo de probe em andamento)."""
    if bool(getattr(self.walker, "e_held_for_mine", False)):
      return True
    return self._fa_sub in ("holding_e", "wait_e", "w", "wait_w")

  def _should_commit_ready(self, hit: Any) -> bool:
    """
    READY com label presente (ou já mining) — unificado com mine-hold:
    - e_held_for_mine → já mining (re-commit / keep)
    - FINAL: só durante holding_e + `_ore_label_present` (found|raw|≥hold_min)
      — sem READY ambient em w/wait_w/wait_e / antes do 1º probe.
    """
    if bool(getattr(self.walker, "e_held_for_mine", False)):
      return True
    if self.phase == Phase.FINAL_APPROACH:
      if self._fa_sub != "holding_e":
        return False
      return self._ore_label_present(hit)
    return self._ore_label_present(hit)

  def _force_stop_movement(self) -> None:
    """Solta W/A/S/D (mantém E se mine-hold). Sem pulse/walk até mining done."""
    self.pursuit.stop_walk()

  def _restart_after_probe_miss(
    self,
    *,
    reason: str,
    score: float = 0.0,
  ) -> tuple[str, str]:
    """
    Pulse-max / probe miss / FINAL timeout: mark_done (soft-avoid nó atual)
    → SCAN + auto-lock — mesmo ciclo que MINING done, outro alvo.
    """
    clear = getattr(self.walker, "clear_e_hold", None)
    if callable(clear):
      clear()
    self.pursuit.stop_walk()
    self.pursuit.mark_done()
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._allow_auto_lock = True
    self._reset_scan_spin()
    self._reset_final_state()
    self._reset_mining_end_state()
    self.phase = Phase.SCAN
    score_s = f" ore={score:.3f}" if score > 0 else ""
    mlog(
      f"[v2] {reason}{score_s} — mark_done | SCAN restart (auto lock)"
    )
    return "probe-miss-restart", "SCAN restart"

  def _enter_close_walk_wait(self) -> tuple[str, str]:
    """
    Close-walk ok → FINAL_APPROACH: probe E, depois até N× (W+wait+E).
    Hit mid-hold → READY; pulse-max miss → mark_done + SCAN restart.
    """
    if self.phase == Phase.FINAL_APPROACH:
      self.pursuit.stop_walk()
      return "idle", "FINAL probe-e"
    if self.phase == Phase.COOLDOWN:
      self.pursuit.stop_walk()
      return "idle", "COOLDOWN aguardando"
    clear = getattr(self.walker, "clear_e_hold", None)
    if callable(clear):
      clear()
    self.pursuit.stop_walk()
    self._allow_auto_lock = False
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._reset_final_state()
    begin = getattr(self.pursuit, "begin_final_approach", None)
    if callable(begin):
      begin()
    self.phase = Phase.FINAL_APPROACH
    mlog(
      f"[v2] CLOSE-WALK done — standstill "
      f"{self.final_wait_before_e_ms:.0f}ms + "
      f"probe E {self.final_probe_e_ms:.0f}ms "
      f"(miss → ≤{self.final_pulse_max}× W+E | hit → mine)"
    )
    return "close-walk-probe", "FINAL probe-e"

  def _ensure_e_mine_hold(self) -> None:
    """
    Garante E fisicamente down + e_held_for_mine.
    Chamar ANTES de stop_walk: senão stop() faz release_all_keys e solta E
    (flag ainda False durante holding_e do FINAL).
    """
    e_held = bool(getattr(self.walker, "e_held_for_mine", False))
    begin = getattr(self.walker, "begin_probe_e", None)
    keep = getattr(self.walker, "keep_e_for_mine", None)
    if not e_held and callable(begin):
      begin()
    if callable(keep):
      keep()

  def _commit_mining_from_final(self, hit: Any) -> tuple[str, str]:
    """
    Ore confirmado no FINAL: garante E down, para movimento, READY mine-hold.
    Chamado só após `_should_commit_ready` (present / e_held).
    """
    score = float(getattr(hit, "score", 0.0) or 0.0)
    # CRÍTICO: E + flag ANTES de stop — holding_e tem E down mas flag False;
    # stop_walk → release_all_keys soltava E e mid-hold só setava o flag.
    mid_hold = self._fa_sub == "holding_e" or bool(
      getattr(self.walker, "e_held_for_mine", False)
    )
    self._ensure_e_mine_hold()
    self._force_stop_movement()
    if mid_hold:
      mlog(f"[v2] Mining ore mid-hold — keep E (ore={score:.3f})")
    else:
      mlog(f"[v2] Mining ore confirmado — E down (ore={score:.3f})")
    return (
      self._enter_ready(reason="MINING_ORE", score=score),
      f"READY ore={score:.3f}",
    )

  def _enter_ready(
    self,
    *,
    reason: str,
    score: float = 0.0,
  ) -> str:
    """Para, limpa lock, READY — Mining ore confirmado; hold E até label sumir."""
    # Sempre E down ao entrar READY (quase→READY / commit / legado).
    self._ensure_e_mine_hold()
    self._force_stop_movement()
    # Mine success: libera nós stuck para re-lock futuro; depois mark_done do atual.
    clear_stuck = getattr(self.pursuit, "clear_stuck_blacklist", None)
    if callable(clear_stuck):
      clear_stuck()
    self.pursuit.mark_done()
    self._allow_auto_lock = False
    self._reset_final_state()
    self._mining_label_seen = True
    # Entrada READY só via label real; peak = score (floor engage só se score=0).
    self._mining_had_strong = True
    self._ore_gone_streak = 0
    self._ore_below_match_streak = 0
    self._ore_peak = max(float(score), float(self.ore_engage_threshold))
    self._ore_score_hist.clear()
    if score > 0:
      self._ore_score_hist.append(float(score))
    self._mining_ready_at = time.perf_counter()
    self.phase = Phase.READY_INTERACT
    score_s = f" ore={score:.3f}" if score > 0 else ""
    mlog(
      f"[v2] READY_INTERACT ({reason}{score_s}) — "
      f"lock limpo | hold E ate Mining ore sumir (F8=proximo)"
    )
    return "ready-interact"

  def _finish_mining(self, *, score: float = 0.0, reason: str = "ore clear") -> str:
    """Label Mining ore sumiu: solta E → SCAN + auto-lock (ciclo completo)."""
    clear = getattr(self.walker, "clear_e_hold", None)
    if callable(clear):
      clear()
    peak = self._ore_peak
    self._reset_mining_end_state()
    # Nó já em `_done_xy` via mark_done em READY — lock_nearest pula o mesmo XY.
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._allow_auto_lock = True
    self._reset_scan_spin()
    self._reset_final_state()
    gate = float(self._post_mine_ore_gate_s)
    if gate > 0:
      self._post_mine_ore_gate_until = time.perf_counter() + gate
    else:
      self._post_mine_ore_gate_until = 0.0
    self.phase = Phase.SCAN
    score_s = f" ore={score:.3f}" if score > 0 else ""
    peak_s = f" peak={peak:.3f}" if peak > 0 else ""
    mlog(
      f"[v2] MINING done — {reason}{score_s}{peak_s} "
      f"(hold≥{self.ore_hold_min:.2f}|gone<{self.ore_gone_threshold:.2f}"
      f"|drop≥{self.ore_gone_drop_from_peak:.2f} below-hold) — "
      f"E up | SCAN restart (auto lock)"
    )
    return "mining-done"

  def _tick_ready_mining(
    self,
    ctx: FrameContext,
    *,
    active: bool,
  ) -> tuple[str, str]:
    """
    Em READY com E held / label vista: poll Mining ore.
    Presente se found/raw_hit, score≥hold_min, ou mediana≥hold_min.
    Absent (conta streak) se score efetivo < gone_threshold, ou já
    abaixo de hold_min com queda soft do peak.
    Timeout: hold longo + below-hold_min, ou hard mine_hold_timeout_s.
    Nunca walk / pulse-W / stuck-D aqui — só hold E até label sumir.
    """
    if self.phase == Phase.COOLDOWN:
      return "idle", "COOLDOWN"

    hit = self._detect_mining_ore(ctx)
    score = float(hit.score)
    thr = float(self.mining_ore.threshold)
    raw_hit = bool(getattr(hit, "raw_hit", False))
    found = bool(hit.found)
    strong = self._ore_label_present(hit)
    absent = self._ore_frame_absent(
      score=score,
      thr=thr,
      raw_hit=raw_hit,
      found=found,
    )

    if strong:
      self._mining_label_seen = True
      self._mining_had_strong = True

    now = time.perf_counter()
    if self._mining_ready_at is None:
      self._mining_ready_at = now
    hold_s = now - self._mining_ready_at

    # Soft-timeout streak: abaixo da banda de hold (não match).
    if score < self.ore_hold_min:
      self._ore_below_match_streak += 1
    else:
      self._ore_below_match_streak = 0

    # Hard timeout: FP platô ≥hold_min nunca ausenta via gone.
    hard_t = self.ore_mine_hold_timeout_s
    if (
      active
      and hard_t > 0
      and hold_s >= hard_t
      and self._mining_label_seen
      and self._mining_had_strong
    ):
      self._force_stop_movement()
      return (
        self._finish_mining(
          score=score,
          reason=f"hold-timeout {hold_s:.1f}s",
        ),
        "SCAN restart",
      )

    # Soft timeout: hold longo com score abaixo de hold_min por confirm frames.
    soft_t = self.ore_mine_hold_below_match_s
    need = self.ore_gone_confirm_frames
    if (
      active
      and soft_t > 0
      and hold_s >= soft_t
      and self._ore_below_match_streak >= need
      and self._mining_label_seen
      and self._mining_had_strong
    ):
      self._force_stop_movement()
      return (
        self._finish_mining(
          score=score,
          reason=f"below-hold {hold_s:.1f}s",
        ),
        "SCAN restart",
      )

    if not absent:
      self._mining_label_seen = True
      self._ore_gone_streak = 0
      # Ore presente: E down ANTES de stop (flag protege release_all_keys).
      self._ensure_e_mine_hold()
      self._force_stop_movement()
      return "mine-hold", f"READY mining ore={score:.3f}"

    # Ore ausente / fraco: para W; E só solta em _finish_mining.
    self._force_stop_movement()

    if not self._mining_label_seen or not self._mining_had_strong:
      return "idle", "READY_INTERACT"

    if not active:
      # Sem foco: não conta desaparecimento (evita falso fim).
      return "idle", f"READY ore-wait (no-focus) ore={score:.3f}"

    self._ore_gone_streak += 1
    if self._ore_gone_streak >= need:
      return (
        self._finish_mining(score=score),
        "SCAN restart",
      )
    return (
      "mine-wait-gone",
      f"READY ore-weak {self._ore_gone_streak}/{need} "
      f"ore={score:.3f} peak={self._ore_peak:.3f}",
    )

  def _abort_final_restart(
    self,
    *,
    reason: str,
    score: float = 0.0,
    dist_px: float | None = None,
  ) -> str:
    """
    Cancela final-approach sem READY.
    Se ainda perto do lock (dist ≤ final_abort_retry_px): reinicia FINAL
    no mesmo alvo. Senão: limpa lock (sem mark_done) e SCAN + auto-lock.
    """
    clear = getattr(self.walker, "clear_e_hold", None)
    if callable(clear):
      clear()
    self.pursuit.stop_walk()

    lock = self.pursuit.v1_lock
    if dist_px is None and lock is not None:
      dist_px = float(getattr(lock, "last_distance_px", 999.0) or 999.0)
    close = (
      lock is not None
      and dist_px is not None
      and float(dist_px) <= float(self.final_abort_retry_px)
    )
    if close:
      self._announced_final = False
      self._reset_final_state()
      begin = getattr(self.pursuit, "begin_final_approach", None)
      if callable(begin):
        begin()
      self.phase = Phase.FINAL_APPROACH
      score_s = f" ore={score:.3f}" if score > 0 else ""
      mlog(
        f"[v2] FINAL-APPROACH abort: {reason}{score_s} — "
        f"retry same target (dist={float(dist_px):.0f}px)"
      )
      return "final-abort-retry"

    self.pursuit.reset()
    self._allow_auto_lock = True
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._reset_final_state()
    self._reset_scan_spin()
    self.phase = Phase.SCAN
    score_s = f" ore={score:.3f}" if score > 0 else ""
    mlog(f"[v2] FINAL-APPROACH abort: {reason}{score_s} — restart")
    return "final-abort-restart"

  def _handle_stuck_idle(self, *, dist_px: float) -> str:
    """
    Path blocked no GOTO walk.

    Até stuck_d_max_attempts: solta W, segura D, arma realign pós-D
    (próximos ticks: câmera até |brg| ok, depois W — mesmo lock).
    Depois: force-avoid XY e re-SCAN (outro lock).

    Nunca invocar em READY / FINAL / e_held / ore forte — caller gateia.
    """
    # Defesa: se E ainda held (mine), não strafe D.
    if getattr(self.walker, "e_held_for_mine", False) is True:
      self.pursuit.stop_walk()
      reset_idle = getattr(self.pursuit, "_reset_stuck_idle", None)
      if callable(reset_idle):
        reset_idle()
      return "stuck-skip-mining"
    attempts = int(self.pursuit._stuck_d_attempts)
    max_att = int(self.pursuit.stuck_d_max_attempts)
    if attempts < max_att:
      self.pursuit.stop_walk()
      action = self.pursuit.recover_stuck_d()
      n = int(self.pursuit._stuck_d_attempts)
      mlog(
        f"[v2] STUCK/IDLE — D recover {n}/{max_att} "
        f"({self.pursuit.stuck_d_hold_ms:.0f}ms) dist={dist_px:.0f}px "
        f"— next realign then W"
      )
      return str(action) if str(action).startswith("strafe-d") else "stuck-idle-d-recover"

    self.pursuit.stop_walk()
    self.pursuit.mark_stuck(facing_deg=self.pursuit._smooth_facing)
    self.pursuit._reset_stuck_d_attempts()
    self._allow_auto_lock = True
    self._announced_fine_align = False
    self._announced_close_walk = False
    self._announced_final = False
    self._reset_final_state()
    self._reset_mining_end_state()
    self._reset_scan_spin()
    self.phase = Phase.SCAN
    mlog(
      f"[v2] STUCK/IDLE — path blocked after {max_att}× D, next lock "
      f"(dist={dist_px:.0f}px)"
    )
    self._arm_lock_cooldown()
    return "stuck-idle-next"

  def _detect_mining_ore(self, ctx: FrameContext) -> Any:
    """Reusa hit do perceive (mesmo detector); senao detecta no HUD."""
    if "mining_ore_score" in ctx.meta:
      return self.mining_ore.last_hit
    return self.mining_ore.detect(ctx.hud_bgr)

  def _final_standstill_before_e(
    self,
    *,
    now: float,
    score: float,
  ) -> tuple[str, str] | None:
    """
    Garante parado + wait final_wait_before_e_ms antes do keydown E.
    Retorna (action, nav) se ainda a esperar; None quando pode começar o probe.
    """
    self.pursuit.stop_walk()
    wait_ms = float(self.final_wait_before_e_ms)
    if wait_ms <= 0:
      return None
    waited_ms = (now - self._fa_sub_at) * 1000.0
    if waited_ms < wait_ms:
      nav = (
        f"FINAL wait-before-e ore={score:.3f} "
        f"[{waited_ms:.0f}/{wait_ms:.0f}ms "
        f"pulses={self._pulse_count}/{self.final_pulse_max}]"
      )
      return "probe-e-wait", nav
    return None

  def _final_probe_e_start(
    self,
    ctx: FrameContext,
    *,
    now: float,
    score: float,
  ) -> tuple[str, str]:
    """Keydown E → holding_e (poll mid-hold no próximo tick)."""
    begin = getattr(self.walker, "begin_probe_e", None)
    if callable(begin):
      action = begin()
    else:
      action = self.walker.probe_e(hold_ms=self.final_probe_e_ms)
      hit2 = self._detect_mining_ore(ctx)
      if self._ore_label_present(hit2):
        self._fa_sub = "holding_e"
        self._fa_sub_at = now
        return self._commit_mining_from_final(hit2)
      # Legacy blocking miss: segue para W se ainda há pulsos.
      if self._pulse_count >= self.final_pulse_max:
        s2 = float(getattr(hit2, "score", 0.0) or 0.0)
        return self._restart_after_probe_miss(
          reason=(
            f"probe E miss pulses="
            f"{self._pulse_count}/{self.final_pulse_max}"
          ),
          score=s2,
        )
      self._fa_sub = "w"
      self._fa_sub_at = now
      return "probe-e-miss", (
        f"FINAL probe-e ore={float(getattr(hit2, 'score', 0.0) or 0.0):.3f} "
        f"[w after miss]"
      )
    self._fa_sub = "holding_e"
    self._fa_sub_at = now
    nav = (
      f"FINAL probe-e ore={score:.3f} "
      f"[holding_e 0/{self.final_probe_e_ms:.0f}ms "
      f"pulses={self._pulse_count}/{self.final_pulse_max}]"
    )
    return action, nav

  def _tick_final_approach(
    self,
    ctx: FrameContext,
    *,
    active: bool,
    pursuit: Any,
  ) -> tuple[str, str]:
    """
    Pós close-walk: probe E inicial; miss → ≤N× (W + wait + E).
    Present só mid-hold E → READY; pulse-max miss → mark_done + SCAN.
    Retorna (action, nav_status).
    """
    _ = pursuit
    now = time.perf_counter()
    if self._final_started_at is None:
      self._final_started_at = now
      self._fa_sub = "e"
      self._fa_sub_at = now
      self.pursuit.stop_walk()

    if not self._announced_final:
      mlog(
        f"[v2] FINAL-APPROACH — standstill "
        f"{self.final_wait_before_e_ms:.0f}ms + "
        f"probe E {self.final_probe_e_ms:.0f}ms "
        f"(miss → ≤{self.final_pulse_max}× "
        f"W {self.final_pulse_w_ms:.0f}ms + "
        f"wait {self.final_wait_after_w_ms:.0f}ms + "
        f"standstill + E)"
      )
      self._announced_final = True

    if not active:
      self.pursuit.stop_walk()
      return "no-focus", "FINAL probe-e"

    hit = self._detect_mining_ore(ctx)
    score = float(hit.score)
    engage = float(self.ore_engage_threshold)
    nav = (
      f"FINAL probe-e ore={score:.3f}/{engage:.2f} "
      f"[{self._fa_sub} pulses={self._pulse_count}/{self.final_pulse_max}]"
    )
    # Quase-engage (abaixo do floor mas perto) ou near_miss do detector.
    near_engage = (not self._ore_engage(hit)) and score >= max(
      0.0, engage - 0.08
    )
    if (
      near_engage or getattr(hit, "near_miss", False)
    ) and self._fa_sub in ("e", "holding_e"):
      prev = getattr(self, "_ore_near_logged", None)
      if (
        prev is None
        or abs(score - prev[0]) >= 0.03
        or (now - prev[1]) >= 1.0
      ):
        mlog(
          f"[v2] Mining ore quase: ore={score:.3f} need>={engage:.2f} "
          f"[{self._fa_sub}]"
        )
        self._ore_near_logged = (score, now)

    # READY só mid-hold E (ou e_held) — para W, keep E.
    if self._should_commit_ready(hit):
      return self._commit_mining_from_final(hit)

    # Label present fora de holding_e: nunca W — começa/retoma probe E.
    if self._fa_sub in ("w", "wait_w", "wait_e") and self._ore_label_present(
      hit
    ):
      begin = getattr(self.walker, "begin_probe_e", None)
      if callable(begin):
        begin()
      self._fa_sub = "holding_e"
      self._fa_sub_at = now
      return self._commit_mining_from_final(hit)

    elapsed = now - (self._final_started_at or now)
    timed_out = (
      self.final_approach_timeout_s > 0
      and elapsed >= self.final_approach_timeout_s
    )
    if timed_out:
      return self._restart_after_probe_miss(
        reason="FINAL probe timeout",
        score=score,
      )

    if self._fa_sub == "e":
      waiting = self._final_standstill_before_e(now=now, score=score)
      if waiting is not None:
        return waiting
      return self._final_probe_e_start(ctx, now=now, score=score)

    if self._fa_sub == "holding_e":
      held_ms = (now - self._fa_sub_at) * 1000.0
      if self._should_commit_ready(hit):
        return self._commit_mining_from_final(hit)
      if held_ms < self.final_probe_e_ms:
        nav = (
          f"FINAL probe-e ore={score:.3f} "
          f"[holding_e {held_ms:.0f}/{self.final_probe_e_ms:.0f}ms "
          f"pulses={self._pulse_count}/{self.final_pulse_max}]"
        )
        return "probe-e-hold", nav
      # Sem Mining ore no prazo → keyup E.
      end = getattr(self.walker, "end_probe_e", None)
      if callable(end):
        end()
      # Já usou todos os W+E cycles → mark_done + SCAN (outro nó).
      if self._pulse_count >= self.final_pulse_max:
        return self._restart_after_probe_miss(
          reason=(
            f"probe E miss pulses="
            f"{self._pulse_count}/{self.final_pulse_max}"
          ),
          score=score,
        )
      # Ainda há ciclos: W → wait → E de novo.
      self._fa_sub = "w"
      self._fa_sub_at = now
      nav = (
        f"FINAL probe-e ore={score:.3f} "
        f"[w next pulses={self._pulse_count}/{self.final_pulse_max}]"
      )
      return "probe-e-miss", nav

    if self._fa_sub == "w":
      # Nunca W com label present (já tratado acima).
      pulse = getattr(self.walker, "pulse_forward", None)
      hold = float(self.final_pulse_w_ms)
      if callable(pulse):
        action = pulse(hold_ms=hold)
      else:
        action = f"pulse-w-{hold:.0f}ms"
      self._pulse_count += 1
      self._fa_sub = "wait_w"
      self._fa_sub_at = now
      mlog(
        f"[v2] FINAL pulse-W {hold:.0f}ms "
        f"({self._pulse_count}/{self.final_pulse_max}) "
        f"ore={score:.3f}"
      )
      nav = (
        f"FINAL pulse-w ore={score:.3f} "
        f"[{self._pulse_count}/{self.final_pulse_max}]"
      )
      return action, nav

    if self._fa_sub == "wait_w":
      waited_ms = (now - self._fa_sub_at) * 1000.0
      if waited_ms < self.final_wait_after_w_ms:
        nav = (
          f"FINAL wait-w ore={score:.3f} "
          f"[{waited_ms:.0f}/{self.final_wait_after_w_ms:.0f}ms "
          f"pulses={self._pulse_count}/{self.final_pulse_max}]"
        )
        return "pulse-w-wait", nav
      # Após wait_w: standstill (final_wait_before_e_ms) → probe E.
      self._fa_sub = "e"
      self._fa_sub_at = now
      waiting = self._final_standstill_before_e(now=now, score=score)
      if waiting is not None:
        return waiting
      return self._final_probe_e_start(ctx, now=now, score=score)

    # Estado inesperado → mark_done + SCAN (não fica parado).
    return self._restart_after_probe_miss(
      reason="FINAL probe idle",
      score=score,
    )

  def tick(
    self,
    ctx: FrameContext,
    *,
    enabled: bool,
    game_focus: bool,
    request_next: bool = False,
  ) -> FrameContext:
    action = "idle"
    scan = self._scan(ctx)
    legacy = ctx.meta.get("legacy_arrow")

    if legacy is not None:
      self.pursuit.update_facing(ctx.arrow.facing_deg)

    if request_next:
      self.pursuit.mark_done()
      self._advance_after_node(ctx, reason="Proximo alvo")

    pursuit = (
      self.pursuit.evaluate(
        scan,
        legacy_arrow=legacy,
        facing_deg=ctx.arrow.facing_deg,
        arrow=ctx.arrow,
      )
      if scan is not None and legacy is not None
      else None
    )

    lock = None
    bearing = None
    dist_px = 0.0
    aligned = False
    arrived = False
    target_dot = None
    nav_status = ""
    active = enabled and game_focus
    move_phase = pursuit.move_phase if pursuit else ""

    if not enabled:
      self.pursuit.stop_walk()
      self._reset_scan_spin()
      action = "idle"

    elif not game_focus:
      self.pursuit.stop_walk()
      action = "no-focus"

    elif self.phase == Phase.SCAN:
      # F6 arm / pós-mine / abort: auto lock_nearest; sem nós → SCAN_SPIN.
      if self._allow_auto_lock:
        action, nav_status = self._tick_scan_or_spin(
          scan, ctx, active=active
        )
        if action == "scan-goto":
          pursuit = (
            self.pursuit.evaluate(
              scan,
              legacy_arrow=legacy,
              facing_deg=ctx.arrow.facing_deg,
              arrow=ctx.arrow,
            )
            if scan is not None and legacy is not None
            else None
          )
          move_phase = pursuit.move_phase if pursuit else ""
      else:
        self._reset_scan_spin()
        self.pursuit.stop_walk()
        action = "scan"

    elif self.phase == Phase.GOTO:
      # STILL_MINING só com e_held + label — nunca ore ambient no fine-align.
      busy, hit = self._mining_busy(ctx)
      if busy and self._can_resume_still_mining(hit):
        action, nav_status = self._resume_ready_still_mining(hit)
        move_phase = "idle"
      elif busy:
        action = self.pursuit.stop_walk()
        nav_status = "GOTO hold (e held, label absent)"
        move_phase = "idle"
      else:
        move = pursuit.move_phase if pursuit else ""
        if pursuit is not None and move in (
          "close_done",
          "final_approach",
          "aligned",
        ):
          # Close-walk (ou legado aligned/final) → probe E único (mesmo tick).
          action, nav_status = self._enter_close_walk_wait()
          if self.phase == Phase.FINAL_APPROACH:
            action, nav_status = self._tick_final_approach(
              ctx, active=active, pursuit=pursuit
            )
            move_phase = "final_approach"
          else:
            move_phase = "idle"
        elif pursuit is not None and move == "close_walk":
          if not self._announced_close_walk:
            need = float(self.pursuit.close_walk_px)
            mlog(
              f"[v2] CLOSE-WALK (sep={pursuit.dist_px:.0f}px) — "
              f"progresso 0/{need:g}px blip→pivot"
            )
            self._announced_close_walk = True
          if active:
            action = self.pursuit.walk(
              pursuit.bearing_deg,
              dist_px=pursuit.dist_px,
              target_dot=pursuit.target_dot,
              close_walk=True,
            )
          else:
            action = self.pursuit.stop_walk()
        elif pursuit is not None and move == "fine_align":
          if not self._announced_fine_align:
            brg = pursuit.bearing_deg
            brg_s = f" brg={brg:+.1f}" if brg is not None else ""
            mlog(
              f"[v2] PAROU (dist={pursuit.dist_px:.0f}px{brg_s}) — "
              f"fine-align camera"
            )
            self._announced_fine_align = True
          if active and pursuit.bearing_deg is not None:
            action = self.pursuit.walk(
              pursuit.bearing_deg,
              dist_px=pursuit.dist_px,
              target_dot=pursuit.target_dot,
              fine_align=True,
            )
          else:
            action = self.pursuit.stop_walk()
        elif pursuit is None:
          # Frame sem scan/arrow — NÃO demote a SCAN nem limpa lock.
          # Demote antigo deixava v1_lock órfão + display congelado + SCAN_SPIN.
          action = self.pursuit.stop_walk()
          nav_status = "GOTO hold (no scan/arrow)"
        elif pursuit.target is None:
          # Ghost abandon já limpou v1_lock; se sobrou órfão, limpa agora.
          if self.pursuit.v1_lock is not None:
            self.pursuit.reset()
          self.phase = Phase.SCAN
          self._allow_auto_lock = True
          self._arm_lost_target_recovery()
          self.pursuit.stop_walk()
          mlog("[v2] GOTO — lost target, re-SCAN (cooldown + spin once)")
          action = "lost-target"
          nav_status = "SCAN lost-target"
        elif active and pursuit.bearing_deg is not None:
          action = self.pursuit.walk(
            pursuit.bearing_deg,
            dist_px=pursuit.dist_px,
            target_dot=pursuit.target_dot,
          )
          # Pós D: realign obrigatório (sem W) — não conta como walk p/ stuck.
          # realign-* normal (rumo grande) NÃO é STUCK — só pós recover_stuck_d.
          if self.pursuit._need_post_stuck_realign:
            expecting_walk = False
            nav_status = "STUCK/IDLE realign after D"
            move_phase = "stuck_d_realign"
          elif str(action).startswith("realign"):
            expecting_walk = False
          else:
            expecting_walk = str(action).startswith("walk")
          if self.pursuit.check_stuck_idle(
            pursuit.dist_px,
            bearing_deg=pursuit.bearing_deg,
            move_phase=move or "",
            expecting_walk=expecting_walk,
          ):
            action = self._handle_stuck_idle(dist_px=pursuit.dist_px)
            if self.phase == Phase.SCAN:
              nav_status = "STUCK/IDLE path blocked"
              move_phase = "stuck_idle"
            elif action == "stuck-skip-mining":
              nav_status = "GOTO hold (mining)"
              move_phase = "idle"
            else:
              nav_status = "STUCK/IDLE D recover"
              move_phase = "stuck_d_recover"
        else:
          action = self.pursuit.stop_walk()
          self.pursuit.check_stuck_idle(
            pursuit.dist_px if pursuit else 0.0,
            bearing_deg=pursuit.bearing_deg if pursuit else None,
            move_phase=move or "",
            expecting_walk=False,
          )

    elif self.phase == Phase.FINAL_APPROACH:
      action, nav_status = self._tick_final_approach(
        ctx, active=active, pursuit=pursuit
      )
      move_phase = "final_approach"
      if pursuit is not None:
        # Mantém métricas do lock enquanto final approach.
        pass

    elif self.phase in (Phase.READY_INTERACT, Phase.COOLDOWN):
      # READY: poll Mining ore → SCAN restart.
      # COOLDOWN: legado idle (probe miss agora vai direto a SCAN).
      self.pursuit.stop_walk()
      if self.phase == Phase.READY_INTERACT:
        action, nav_status = self._tick_ready_mining(ctx, active=active)
      else:
        clear = getattr(self.walker, "clear_e_hold", None)
        if callable(clear):
          clear()
        self._allow_auto_lock = False
        action = "idle"
        nav_status = nav_status or "COOLDOWN aguardando"

    else:
      # Fases antigas (INTERACT/MINING) → idle neste teste.
      self.phase = Phase.READY_INTERACT
      self.pursuit.stop_walk()
      action = "idle"

    # Uma unica avaliacao por tick.
    # SCAN / READY: nunca publicar lock órfão (regressão linha verde congelada).
    # COOLDOWN pós close-walk: mantém lock visível para planejar.
    if pursuit is not None and self.phase not in (
      Phase.SCAN,
      Phase.READY_INTERACT,
    ):
      lock = pursuit.display_lock
      bearing = pursuit.bearing_deg
      dist_px = pursuit.dist_px
      aligned = pursuit.aligned
      arrived = pursuit.arrived
      if not nav_status:
        nav_status = pursuit.nav_status
      target_dot = pursuit.target_dot
      if not move_phase:
        move_phase = pursuit.move_phase
    elif (
      self.pursuit.v1_lock is not None
      and self.phase
      in (Phase.GOTO, Phase.FINAL_APPROACH, Phase.COOLDOWN)
    ):
      # Fallback só em navegação ativa: scan/arrow ausente neste frame.
      v1 = self.pursuit.v1_lock
      lock = V2TargetLock(
        track_id=int(v1.node_id or 0),
        x=float(v1.locked_x),
        y=float(v1.locked_y),
        tier=str(v1.tier).lower(),
        lost_frames=int(v1.lost_frames),
        pinned=True,
      )

    ore_score = self.mining_ore.last_score
    ore_found = bool(self.mining_ore.last_hit.found)
    if "mining_ore_score" in ctx.meta:
      ore_score = float(ctx.meta.get("mining_ore_score") or ore_score)
      ore_found = bool(ctx.meta.get("mining_ore_found", ore_found))

    return ctx.with_updates(
      lock=lock,
      phase=self.phase,
      bearing_deg=bearing,
      dist_px=dist_px,
      aligned=aligned,
      arrived=arrived,
      action=action,
      meta={
        **ctx.meta,
        "target_dot": target_dot,
        "nav_status": nav_status,
        "move_phase": move_phase,
        "mining_ore_score": ore_score,
        "mining_ore_found": ore_found,
        "final_pulses": self._pulse_count,
      },
    )
