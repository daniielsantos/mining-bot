"""Unit tests for Mining ore ROI two-corner calibration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from v2.ore_roi_calibrate import (  # noqa: E402
  apply_mining_ore_roi,
  calibrate_display_size,
  map_preview_click_to_screen,
  mining_ore_roi_patch,
  place_roi_at_click,
  place_roi_from_clicks,
  render_ore_calibrate_view,
  roi_centered_on,
  roi_from_corners,
)
import numpy as np  # noqa: E402


def test_roi_centered_on_clamps_edges() -> None:
  roi = roi_centered_on(10, 10, width=1000, height=100, screen_w=2560, screen_h=1440)
  assert roi["left"] == 0
  assert roi["top"] == 0
  assert roi["width"] == 1000
  assert roi["height"] == 100


def test_roi_centered_on_mid_screen() -> None:
  roi = roi_centered_on(1280, 900, width=1000, height=100, screen_w=2560, screen_h=1440)
  assert roi == {"left": 780, "top": 850, "width": 1000, "height": 100}


def test_roi_from_corners_any_order() -> None:
  a = roi_from_corners(1000, 1200, 1520, 1255, screen_w=2560, screen_h=1440)
  b = roi_from_corners(1520, 1255, 1000, 1200, screen_w=2560, screen_h=1440)
  assert a == b
  assert a == {"left": 1000, "top": 1200, "width": 521, "height": 56}


def test_roi_from_corners_enforces_min_size() -> None:
  roi = roi_from_corners(100, 100, 105, 105, screen_w=2560, screen_h=1440)
  assert roi["width"] >= 80
  assert roi["height"] >= 24


def test_map_preview_click_to_screen() -> None:
  # 2560x1440 fit to width 380 → scale = 380/2560 (igual fit_width)
  scale = 380 / 2560.0
  mapped = map_preview_click_to_screen(
    int(1280 * scale),
    int(900 * scale),
    frame_hw=(1440, 2560),
    display_width=380,
  )
  assert mapped is not None
  sx, sy = mapped
  assert abs(sx - 1280) <= 5
  assert abs(sy - 900) <= 5


def test_map_outside_returns_none() -> None:
  assert (
    map_preview_click_to_screen(
      10,
      5000,
      frame_hw=(1440, 2560),
      display_width=380,
    )
    is None
  )


def test_map_large_calibrate_window() -> None:
  # Janela grande ~1600x900 para 2560x1440
  dw, dh, scale = calibrate_display_size(2560, 1440, max_w=1600, max_h=900)
  assert dw == 1600
  assert dh == 900
  assert abs(scale - (1600 / 2560)) < 1e-6
  mapped = map_preview_click_to_screen(
    int(1280 * scale),
    int(900 * scale),
    frame_hw=(1440, 2560),
    display_width=dw,
    display_height=dh,
  )
  assert mapped is not None
  sx, sy = mapped
  assert abs(sx - 1280) <= 3
  assert abs(sy - 900) <= 3


def test_render_ore_calibrate_view_size() -> None:
  frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
  roi = {"left": 1000, "top": 1200, "width": 520, "height": 56}
  view = render_ore_calibrate_view(
    frame,
    roi,
    display_wh=(1600, 900),
  )
  assert view.shape[1] == 1600
  assert view.shape[0] == 900


def test_place_and_apply_uses_current_size() -> None:
  cfg = {
    "resolution": {"width": 2560, "height": 1440},
    "screen_roi": {"left": 100, "top": 100, "width": 800, "height": 80},
    "navigation": {
      "mining_ore": {"roi": {"left": 100, "top": 100, "width": 800, "height": 80}}
    },
  }
  roi = place_roi_at_click(cfg, 1280, 900)
  assert roi["width"] == 800
  assert roi["height"] == 80
  assert roi["left"] == 880  # 1280 - 400
  assert roi["top"] == 860  # 900 - 40
  applied = apply_mining_ore_roi(cfg, roi)
  assert cfg["screen_roi"] == applied
  assert cfg["navigation"]["mining_ore"]["roi"] == applied


def test_place_from_clicks_saves_exact_box() -> None:
  cfg = {
    "resolution": {"width": 2560, "height": 1440},
    "screen_roi": {"left": 0, "top": 0, "width": 1000, "height": 100},
    "navigation": {"mining_ore": {"roi": {"left": 0, "top": 0, "width": 1000, "height": 100}}},
  }
  roi = place_roi_from_clicks(cfg, 900, 1280, 1400, 1330)
  apply_mining_ore_roi(cfg, roi)
  assert cfg["navigation"]["mining_ore"]["roi"] == {
    "left": 900,
    "top": 1280,
    "width": 501,
    "height": 51,
  }
  assert cfg["screen_roi"] == cfg["navigation"]["mining_ore"]["roi"]


def test_patch_keys() -> None:
  patch = mining_ore_roi_patch({"left": 1, "top": 2, "width": 3, "height": 4})
  assert patch["screen_roi"]["left"] == 1
  assert patch["navigation"]["mining_ore"]["roi"]["height"] == 4


if __name__ == "__main__":
  test_roi_centered_on_clamps_edges()
  test_roi_centered_on_mid_screen()
  test_roi_from_corners_any_order()
  test_roi_from_corners_enforces_min_size()
  test_map_preview_click_to_screen()
  test_map_outside_returns_none()
  test_map_large_calibrate_window()
  test_render_ore_calibrate_view_size()
  test_place_and_apply_uses_current_size()
  test_place_from_clicks_saves_exact_box()
  test_patch_keys()
  print("ok")
