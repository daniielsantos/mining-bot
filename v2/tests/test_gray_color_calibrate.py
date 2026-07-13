"""Offline: gray HSV calibration respects config bounds (no road-wide expand)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from v2.vendor.node_detector import MiningNodeDetector  # noqa: E402
from v2.color_calibrate import pick_gray_node_hsv  # noqa: E402


def _det(lower, upper, *, expand: bool = False) -> MiningNodeDetector:
  return MiningNodeDetector(
    tier_colors_hsv={"gray": {"lower": list(lower), "upper": list(upper)}},
    allowed_tiers=["gray"],
    road_gray_range=[75, 150],
    gray_achromatic_v_min=160,
    gray_achromatic_s_max=90,
    gray_achromatic_expand=expand,
    road_bright_protect_min=200,
    center_exclusion_radius_px=2.0,
  )


def test_tier_mask_uses_calibrated_bounds_only() -> None:
  """With expand=False, wide gray_achromatic_* must not reopen mid-gray roads."""
  frame = np.zeros((40, 40, 3), dtype=np.uint8)
  frame[:] = (175, 175, 178)  # mid-gray road-ish BGR
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  det = _det([0, 0, 200], [179, 40, 255], expand=False)
  mask = det._tier_mask(hsv, "gray")
  assert int(mask.max()) == 0, "road matched despite calibrated V_min=200"


def test_expand_opt_in_can_widen() -> None:
  frame = np.zeros((40, 40, 3), dtype=np.uint8)
  frame[:] = (210, 210, 210)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  det = _det([0, 0, 185], [179, 10, 255], expand=True)
  mask = det._tier_mask(hsv, "gray")
  assert int(mask.max()) == 255


def test_pick_gray_stays_above_road_band() -> None:
  frame = np.zeros((60, 60, 3), dtype=np.uint8)
  frame[:] = (30, 30, 30)
  cv2.circle(frame, (30, 30), 5, (245, 245, 245), -1)
  lower, upper, sample = pick_gray_node_hsv(frame, 30, 30)
  assert sample[2] >= 200
  assert lower[2] >= 170, f"V_min too low (roads): {lower}"
  assert upper[1] <= 70, f"S_max too wide: {upper}"
  assert lower[0] == 0 and upper[0] == 179


def test_bright_disk_still_detected() -> None:
  frame = np.zeros((80, 80, 3), dtype=np.uint8)
  frame[:] = (22, 24, 22)
  cv2.rectangle(frame, (10, 50), (70, 62), (175, 175, 178), -1)
  true = (40, 25)
  cv2.circle(frame, true, 4, (240, 240, 242), -1)
  det = _det([0, 0, 185], [179, 70, 255], expand=False)
  scan = det.scan_blips(frame, player_x=40.0, player_y=70.0, min_distance_px=8.0)
  assert scan.nodes
  for node in scan.nodes:
    assert abs(node.y - 56) > 6.0, f"road as node: {node}"


if __name__ == "__main__":
  test_tier_mask_uses_calibrated_bounds_only()
  test_expand_opt_in_can_widen()
  test_pick_gray_stays_above_road_band()
  test_bright_disk_still_detected()
  print("ok")
