"""
Mining Bot v2 — blip-only (standalone).

  cd mining_bot/v2          # ou copie só esta pasta
  pip install -r requirements.txt
  python -m v2.main --preview
  python main.py --preview   # alternativa (mesma pasta v2/)

Controles: F6 liga/desliga | F7 pausa | F9 sai | F8 proximo alvo
  E = proximo so em idle (READY/SCAN/…) — nunca no FINAL_APPROACH (probe Mining ore).

Calibrar cor do nó cinza (preview):
  G  — arma o próximo clique
  clique esquerdo no disco branco do nó no minimapa
  → salva tier_colors_hsv.gray em v2/config.json

Calibrar ROI Mining ore (preview):
  O ou M — abre janela grande "Calibrar Mining ore ROI"
  1º clique = canto superior-esquerdo da barra escura
  2º clique = canto inferior-direito (ou arraste TL→BR)
  → grava navigation.mining_ore.roi (+ screen_roi) e aplica ao vivo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT))

import v2.bootstrap as bootstrap

bootstrap.setup()

import cv2
from pynput import keyboard

from v2.vendor.keyboard_input import (
  IS_WINDOWS,
  get_foreground_window_title,
  is_game_foreground,
  release_all_keys,
)
from v2.vendor.logger import close_log, init_log, mlog
from v2.brain.tick import Brain
from v2.capture.grabber import Grabber
from v2.color_calibrate import (
  apply_gray_to_runtime,
  map_preview_click_to_minimap,
  pick_gray_node_hsv,
  save_gray_color,
)
from v2.core.config import get_hud_roi, load_config
from v2.core.legacy import build_perception_stack
from v2.debug.overlay import render_overlay
from v2.debug.recorder import SessionRecorder
from v2.ore_roi_calibrate import (
  ORE_CAL_WINDOW_NAME,
  apply_mining_ore_roi,
  close_ore_calibrate_window,
  map_preview_click_to_screen,
  open_ore_calibrate_window,
  place_roi_from_clicks,
  render_ore_calibrate_view,
  save_mining_ore_roi,
  screen_size,
)
from v2.perception.pipeline import perceive


def main() -> None:
  parser = argparse.ArgumentParser(description="Mining Bot v2 (blip-only)")
  parser.add_argument("--preview", action="store_true")
  parser.add_argument("--config", type=Path, default=None)
  args = parser.parse_args()

  cfg = load_config(args.config)
  log_path = init_log()
  mlog("[v2] F6 | abordagem 1a pessoa | W+camera | close→probe E→Mining ore")
  mlog("[v2] Ao chegar: close-walk → probe E → mine|SCAN | F8=proximo | F9=sair")
  mlog("[v2] E=proximo so fora do FINAL_APPROACH (la E=probe Mining ore)")
  if args.preview:
    mlog("[v2] Cor gray: tecla G + clique no disco branco do nó (salva v2/config.json)")
    mlog(
      "[v2] Ore ROI: tecla O/M → janela grande → 1º clique TL + 2º BR "
      "(ou arraste)"
    )
  if cfg.get("navigation", {}).get("camera", {}).get("align_only"):
    mlog("[v2] Modo align_only — só gira câmera (sem W)")
  else:
    cam = cfg.get("navigation", {}).get("camera", {})
    nav = cfg.get("navigation", {})
    ore = nav.get("mining_ore", {})
    mlog(
      f"[v2] arrive≤{nav.get('arrive_px', 15)}px | "
      f"close_walk={nav.get('close_walk_px', 15.5)}px | "
      f"final_pulse={nav.get('final_pulse_w_ms', 400)}ms "
      f"max={nav.get('final_pulse_max', 30)} | "
      f"probe_e={nav.get('final_probe_e_ms', 750)}ms "
      f"before_e={nav.get('final_wait_before_e_ms', 200)}ms "
      f"ore≥{ore.get('match_threshold', 0.70)} "
      f"(READY=present≥hold {ore.get('hold_min', 0.70)}; "
      f"quase≥{ore.get('engage_threshold', 0.70)}) "
      f"hold≥{ore.get('hold_min', 0.70)}"
      f"|gone<{ore.get('gone_threshold', 0.40)}"
      f"|drop≥{ore.get('gone_drop_from_peak', 0.20)}"
      f"×{ore.get('gone_confirm_frames', 8)} "
      f"hold≤{ore.get('mine_hold_timeout_s', 20.0)}s "
      f"deadband={cam.get('look_deadband_deg', 3.5)}° "
      f"walk_max={cam.get('walk_max_deg', 12)}°"
    )
  gray_spec = cfg.get("tier_colors_hsv", {}).get("gray")
  if gray_spec:
    mlog(
      f"[v2] gray HSV L={gray_spec.get('lower')} U={gray_spec.get('upper')} "
      f"expand={cfg.get('gray_achromatic_expand', False)}"
    )
  hud_roi = get_hud_roi(cfg)
  mlog(
    f"[v2] ore ROI L={hud_roi['left']} T={hud_roi['top']} "
    f"{hud_roi['width']}x{hud_roi['height']}"
  )
  mlog(f"[v2] Log: {log_path}")

  node_detector, arrow_tracker, screen_ui = build_perception_stack(cfg)
  brain = Brain(cfg, node_detector)
  fps = float(cfg.get("capture_fps", 30))
  panel_width = int(cfg.get("debug", {}).get("preview_panel_width", 380))
  yellow_ray_px = float(cfg.get("navigation", {}).get("yellow_ray_px", 12.0))
  ore_overlay_zoom = float(
    cfg.get("navigation", {}).get("mining_ore", {}).get("overlay_zoom", 2.5)
  )
  focus_kw = cfg.get("game_foreground_keywords")
  focus_proc = cfg.get("game_process_names")
  record_in_preview = bool(cfg.get("debug", {}).get("record_in_preview", True))

  recorder = SessionRecorder(
    cfg,
    Path(__file__).resolve().parent
    / str(cfg.get("debug", {}).get("record_dir", "debug_captures_v2")),
  )

  enabled = False
  paused = False
  running = True
  tick_n = 0
  request_next = False
  last_focus_warn = 0.0
  debug_log_at = 0.0
  last_minimap = None
  last_screen = None
  arm_gray_pick = False
  arm_ore_roi = False
  ore_roi_corner1: tuple[int, int] | None = None
  ore_roi_hover: tuple[int, int] | None = None
  ore_roi_drag_start: tuple[int, int] | None = None
  ore_cal_display_wh: tuple[int, int] | None = None

  toggle_key = keyboard.Key.f6
  pause_key = keyboard.Key.f7
  quit_key = keyboard.Key.f9
  switch_key = keyboard.Key.f8

  def game_focus() -> bool:
    return (not IS_WINDOWS) or is_game_foreground(
      keywords=focus_kw,
      process_names=focus_proc,
    )

  def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
    nonlocal enabled, paused, running, request_next
    if key == toggle_key:
      enabled = not enabled
      mlog(f"[v2] {'LIGADO' if enabled else 'DESLIGADO'}")
      if enabled:
        brain.reset_session()
        if not args.preview or not record_in_preview:
          path = recorder.start("F6")
          if path:
            mlog(f"[v2] Gravando → {path}")
      else:
        brain.walker.stop()
        release_all_keys()
        if not args.preview or not record_in_preview:
          recorder.stop()
    elif key == pause_key and enabled:
      paused = not paused
      mlog(f"[v2] {'PAUSADO' if paused else 'RETOMADO'}")
      if paused:
        brain.walker.stop()
    elif key == quit_key:
      running = False
      brain.walker.stop()
    elif key == switch_key:
      # F8 = abort/proximo em qualquer fase (incl. mid FINAL_APPROACH).
      request_next = True
    elif getattr(key, "char", None) and key.char and key.char.lower() == "e":
      # Bot pressiona E no FINAL_APPROACH (probe Mining ore). O listener
      # global pynput captura esse keydown sintetico — NUNCA tratar como
      # "proximo" nessa fase (mesmo com --preview), senao mark_done mid-hold.
      if brain.phase.value == "FINAL_APPROACH":
        pass  # probe E do bot — nao avancar
      elif (not enabled) or brain.phase.value in (
        "SCAN",
        "GOTO",
        "READY_INTERACT",
        "COOLDOWN",
      ):
        request_next = True

  keyboard.Listener(on_press=on_press).start()

  def _reset_ore_cal() -> None:
    nonlocal arm_ore_roi, ore_roi_corner1, ore_roi_hover, ore_roi_drag_start
    nonlocal ore_cal_display_wh
    arm_ore_roi = False
    ore_roi_corner1 = None
    ore_roi_hover = None
    ore_roi_drag_start = None
    ore_cal_display_wh = None
    close_ore_calibrate_window()

  def _open_ore_cal() -> None:
    nonlocal arm_ore_roi, arm_gray_pick, ore_roi_corner1, ore_roi_hover
    nonlocal ore_roi_drag_start, ore_cal_display_wh
    arm_ore_roi = True
    arm_gray_pick = False
    ore_roi_corner1 = None
    ore_roi_hover = None
    ore_roi_drag_start = None
    sw, sh = screen_size(cfg)
    ore_cal_display_wh = open_ore_calibrate_window(sw, sh)
    cv2.setMouseCallback(ORE_CAL_WINDOW_NAME, on_ore_cal_mouse)
    mlog(
      f"[v2/cal] Ore ROI: janela '{ORE_CAL_WINDOW_NAME}' "
      f"{ore_cal_display_wh[0]}x{ore_cal_display_wh[1]} — "
      "1º clique = canto SUPERIOR-ESQUERDO da barra escura; "
      "2º = INFERIOR-DIREITO (ou arraste)"
    )

  def _commit_ore_roi(sx1: int, sy1: int, sx2: int, sy2: int) -> None:
    nonlocal arm_ore_roi, ore_roi_corner1, ore_roi_hover, ore_roi_drag_start
    roi = place_roi_from_clicks(cfg, sx1, sy1, sx2, sy2)
    apply_mining_ore_roi(cfg, roi)
    path = save_mining_ore_roi(roi)
    _reset_ore_cal()
    mlog(
      f"[v2/cal] ore ROI ({sx1},{sy1})→({sx2},{sy2}) → "
      f"L={roi['left']} T={roi['top']} {roi['width']}x{roi['height']} "
      f"salvo em {path}"
    )

  def on_ore_cal_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    nonlocal ore_roi_corner1, ore_roi_hover, ore_roi_drag_start

    if not arm_ore_roi:
      return
    if last_screen is None or ore_cal_display_wh is None:
      if event == cv2.EVENT_LBUTTONDOWN:
        mlog("[v2/cal] Sem frame de tela ainda")
      return

    # Clique no frame escalado da janela dedicada.
    mapped = map_preview_click_to_screen(
      x,
      y,
      frame_hw=last_screen.shape[:2],
      display_width=ore_cal_display_wh[0],
      display_height=ore_cal_display_wh[1],
    )
    if event == cv2.EVENT_MOUSEMOVE:
      if mapped is not None:
        ore_roi_hover = mapped
      return

    if event == cv2.EVENT_LBUTTONDOWN:
      if mapped is None:
        mlog("[v2/cal] Clique fora da imagem — use a barra Mining ore")
        return
      sx, sy = mapped
      if ore_roi_corner1 is None:
        ore_roi_corner1 = (sx, sy)
        ore_roi_drag_start = (sx, sy)
        ore_roi_hover = (sx, sy)
        mlog(
          f"[v2/cal] Canto 1 TL=({sx},{sy}) — "
          "agora clique (ou solte o arraste) no canto BR"
        )
      else:
        _commit_ore_roi(ore_roi_corner1[0], ore_roi_corner1[1], sx, sy)
      return

    if event == cv2.EVENT_LBUTTONUP and ore_roi_drag_start is not None:
      if mapped is None:
        return
      sx, sy = mapped
      dx = abs(sx - ore_roi_drag_start[0])
      dy = abs(sy - ore_roi_drag_start[1])
      # Arraste com tamanho minimo → completa; clique curto → espera 2º clique.
      if dx >= 40 or dy >= 16:
        _commit_ore_roi(
          ore_roi_drag_start[0], ore_roi_drag_start[1], sx, sy
        )
      else:
        ore_roi_drag_start = None
      return

  def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
    nonlocal arm_gray_pick, last_minimap

    # Ore ROI usa a janela dedicada (on_ore_cal_mouse).
    if arm_ore_roi:
      return

    if event != cv2.EVENT_LBUTTONDOWN:
      return

    if not arm_gray_pick:
      return
    if last_minimap is None:
      mlog("[v2/cal] Sem frame de minimapa ainda")
      return
    mapped = map_preview_click_to_minimap(
      x,
      y,
      minimap_hw=last_minimap.shape[:2],
      panel_width=panel_width,
    )
    if mapped is None:
      mlog("[v2/cal] Clique fora do minimapa — clique no disco branco do nó")
      return
    mx, my = mapped
    lower, upper, sample = pick_gray_node_hsv(last_minimap, mx, my)
    apply_gray_to_runtime(cfg, node_detector, lower, upper)
    path = save_gray_color(lower, upper)
    arm_gray_pick = False
    mlog(
      f"[v2/cal] gray sample HSV={list(sample)} → L={lower} U={upper} "
      f"salvo em {path}"
    )

  if args.preview:
    cv2.namedWindow("Mining v2", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Mining v2", on_mouse)
    mlog("[v2] Preview | Q/ESC sai | G + clique = gray | O/M = ore ROI (janela grande)")
    if record_in_preview:
      path = recorder.start("preview")
      if path:
        mlog(f"[v2] Gravando → {path}")

  try:
    with Grabber(cfg) as grabber:
      while running:
        t0 = time.perf_counter()
        minimap, hud = grabber.grab()
        last_minimap = minimap
        tick_n += 1

        do_tick = enabled and not paused

        ctx = perceive(
          tick_n,
          minimap,
          hud,
          arrow_tracker=arrow_tracker,
          node_detector=node_detector,
          screen_ui=screen_ui,
          cfg=cfg,
          phase=brain.phase,
          v1_lock=brain.pursuit.v1_lock,
          mining_ore=brain.mining_ore,
        )

        if args.preview and not enabled:
          brain.try_preview_lock(ctx)

        if do_tick or request_next:
          ctx = brain.tick(
            ctx,
            enabled=do_tick,
            game_focus=game_focus(),
            request_next=request_next,
          )
          request_next = False
        else:
          brain.walker.stop()
          scan = ctx.meta.get("scan")
          legacy = ctx.meta.get("legacy_arrow")
          pursuit = (
            brain.pursuit.evaluate(
              scan,
              legacy_arrow=legacy,
              facing_deg=ctx.arrow.facing_deg,
              arrow=ctx.arrow,
            )
            if scan is not None and legacy is not None
            else None
          )
          ctx = ctx.with_updates(
            lock=pursuit.display_lock if pursuit else None,
            phase=brain.phase,
            bearing_deg=pursuit.bearing_deg if pursuit else None,
            dist_px=pursuit.dist_px if pursuit else 0.0,
            aligned=pursuit.aligned if pursuit else False,
            action="preview" if args.preview else "idle",
            meta={
              **ctx.meta,
              "nav_status": pursuit.nav_status if pursuit else "",
              "arm_gray_pick": arm_gray_pick,
              "arm_ore_roi": arm_ore_roi,
              "ore_roi_corner1": ore_roi_corner1 is not None,
            },
          )

        now = time.perf_counter()
        # Throttle general dbg; always log each W pulse so FINAL N/max is not sampled.
        log_every = float(cfg.get("debug", {}).get("log_interval_s", 2.0))
        is_final_pulse = (
          ctx.phase.value == "FINAL_APPROACH"
          and (
            str(ctx.action).startswith("pulse-w")
            or str(ctx.action).startswith("probe-e")
          )
        )
        if is_final_pulse or now - debug_log_at >= log_every:
          brg = f"{ctx.bearing_deg:+.0f}" if ctx.bearing_deg is not None else "NA"
          nav = ctx.meta.get("nav_status", "")
          mlog(
            f"[v2/dbg] {ctx.phase.value} dist={ctx.dist_px:.0f} "
            f"brg={brg} {ctx.action} {nav}"
          )
          debug_log_at = now

        if enabled and not paused and IS_WINDOWS and not game_focus():
          if now - last_focus_warn > 3.0:
            mlog(f"[v2] Sem foco — clique no GTA ({get_foreground_window_title()})")
            last_focus_warn = now

        panel = None
        if args.preview:
          meta = dict(ctx.meta)
          if arm_gray_pick:
            meta["arm_gray_pick"] = True
          if arm_ore_roi:
            meta["arm_ore_roi"] = True
            if ore_roi_corner1 is not None:
              meta["ore_roi_corner1"] = True
          ctx = ctx.with_updates(meta=meta)

          cal_frame = None
          if arm_ore_roi:
            cal_frame = grabber.grab_screen()
            last_screen = cal_frame
            cal_roi = get_hud_roi(cfg)
            cal_draft = None
            if ore_roi_corner1 is not None and ore_roi_hover is not None:
              cal_draft = place_roi_from_clicks(
                cfg,
                ore_roi_corner1[0],
                ore_roi_corner1[1],
                ore_roi_hover[0],
                ore_roi_hover[1],
              )
            if ore_cal_display_wh is None:
              ore_cal_display_wh = open_ore_calibrate_window(
                cal_frame.shape[1], cal_frame.shape[0]
              )
              cv2.setMouseCallback(ORE_CAL_WINDOW_NAME, on_ore_cal_mouse)
            cal_view = render_ore_calibrate_view(
              cal_frame,
              cal_roi,
              draft_roi=cal_draft,
              has_corner1=ore_roi_corner1 is not None,
              display_wh=ore_cal_display_wh,
            )
            cv2.imshow(ORE_CAL_WINDOW_NAME, cal_view)

          # Preview normal (minimapa); calibração ore fica na janela dedicada.
          panel = render_overlay(
            ctx,
            screen_ui=screen_ui,
            panel_width=panel_width,
            yellow_ray_px=yellow_ray_px,
            enabled=enabled,
            game_focus=game_focus(),
            overlay_zoom=ore_overlay_zoom,
          )
          cv2.imshow("Mining v2", panel)
          key = cv2.waitKey(1) & 0xFF
          if key in (ord("q"), 27):
            if arm_ore_roi or arm_gray_pick:
              _reset_ore_cal()
              arm_gray_pick = False
              mlog("[v2/cal] Calibracao cancelada")
            else:
              running = False
          elif key in (ord("e"), ord("E")):
            # Mesma regra do listener: E ≠ proximo durante FINAL_APPROACH.
            if brain.phase.value != "FINAL_APPROACH":
              request_next = True
          elif key in (ord("g"), ord("G")):
            arm_gray_pick = True
            _reset_ore_cal()
            mlog("[v2/cal] Clique no disco BRANCO do nó cinza no minimapa")
          elif key in (ord("o"), ord("O"), ord("m"), ord("M")):
            if arm_ore_roi:
              _reset_ore_cal()
              mlog("[v2/cal] Ore ROI cancelado")
            else:
              _open_ore_cal()

        if recorder.session_dir is not None and (args.preview or enabled):
          img = panel if panel is not None else minimap
          recorder.maybe_save(
            img,
            ctx,
            enabled=enabled,
            game_focus=game_focus(),
          )

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, (1.0 / fps) - elapsed))

  except KeyboardInterrupt:
    pass
  finally:
    brain.walker.stop()
    release_all_keys()
    recorder.stop()
    close_log()
    if args.preview:
      cv2.destroyAllWindows()
    mlog("[v2] Encerrado.")


if __name__ == "__main__":
  main()
