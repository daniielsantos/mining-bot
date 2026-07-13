"""
Calibra a cor HSV do nó cinza e grava em v2/config.json.

  cd mining_bot
  python -m v2.calibrate_gray

  Clique esquerdo no disco BRANCO do nó (painel esquerdo)
  S — salva (também auto-salva no clique)
  Q / ESC — sair

O detector usa só tier_colors_hsv.gray (sem expandir acromático para estradas).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT))

import v2.bootstrap as bootstrap

bootstrap.setup()

import cv2
import numpy as np

from v2.vendor.display import add_status_bar, fit_width
from v2.vendor.logger import mlog
from v2.capture.grabber import Grabber
from v2.color_calibrate import (
  apply_gray_to_runtime,
  pick_gray_node_hsv,
  save_gray_color,
)
from v2.core.config import load_config
from v2.core.legacy import build_perception_stack


def main() -> None:
  print(__doc__)
  cfg = load_config()
  detector, _arrow, _ui = build_perception_stack(cfg)
  last_frame: np.ndarray | None = None
  pending: tuple[list[int], list[int]] | None = None

  def on_mouse(event: int, x: int, y: int, _f: int, _p: object) -> None:
    nonlocal last_frame, pending
    if event != cv2.EVENT_LBUTTONDOWN or last_frame is None:
      return
    mw = last_frame.shape[1]
    if x >= mw:
      return
    lower, upper, sample = pick_gray_node_hsv(last_frame, x, y)
    apply_gray_to_runtime(cfg, detector, lower, upper)
    pending = (lower, upper)
    path = save_gray_color(lower, upper)
    msg = f"sample HSV={list(sample)} L={lower} U={upper} → {path}"
    print(f"[cal-gray] {msg}")
    mlog(f"[cal-gray] {msg}")

  cv2.namedWindow("Calibrate Gray", cv2.WINDOW_AUTOSIZE)
  cv2.setMouseCallback("Calibrate Gray", on_mouse)

  with Grabber(cfg) as grabber:
    while True:
      frame, _hud = grabber.grab()
      last_frame = frame
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      mask = detector._tier_mask(hsv, "gray")
      dbg = frame.copy()
      ys, xs = np.where(mask > 0)
      if xs.size:
        dbg[ys, xs] = (40, 220, 40)
      # Soft blend so the node core stays visible.
      out = cv2.addWeighted(frame, 0.55, dbg, 0.45, 0)
      panel = np.hstack([out, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
      panel = fit_width(panel, 720)
      spec = cfg.get("tier_colors_hsv", {}).get("gray", {})
      lines = [
        "clique no disco BRANCO do no | S=salvar | Q=sair",
        f"L={spec.get('lower')} U={spec.get('upper')} expand={cfg.get('gray_achromatic_expand', False)}",
      ]
      panel = add_status_bar(panel, lines, bar_height=48)
      cv2.imshow("Calibrate Gray", panel)
      key = cv2.waitKey(1) & 0xFF
      if key in (ord("q"), 27):
        break
      if key in (ord("s"), ord("S")):
        if pending is not None:
          path = save_gray_color(pending[0], pending[1])
          print(f"[cal-gray] salvo {path}")
        else:
          spec = cfg.get("tier_colors_hsv", {}).get("gray")
          if spec:
            path = save_gray_color(list(spec["lower"]), list(spec["upper"]))
            print(f"[cal-gray] regravado {path}")
      time.sleep(0.01)

  cv2.destroyAllWindows()
  mlog("[cal-gray] Encerrado")


if __name__ == "__main__":
  main()
