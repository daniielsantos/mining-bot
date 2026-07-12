"""Offline checks: gray node center vs visual white disk (Y bias)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from node_detector import MiningNodeDetector  # noqa: E402


def _detector(**overrides) -> MiningNodeDetector:
  kwargs = dict(
    tier_colors_hsv={
      "gray": {"lower": [0, 0, 185], "upper": [179, 70, 255]},
    },
    allowed_tiers=["gray"],
    road_gray_range=[75, 150],
    min_blob_area=2,
    max_blob_area=70,
    min_circularity=0.55,
    max_aspect_ratio=1.55,
    min_solidity=0.75,
    max_enclosing_radius_px=9.0,
    center_exclusion_radius_px=2.0,
    gray_achromatic_v_min=185,
    gray_achromatic_s_max=70,
    gray_achromatic_expand=False,
    road_bright_protect_min=200,
  )
  kwargs.update(overrides)
  return MiningNodeDetector(**kwargs)


def test_half_disk_moments_bias_enclosing_recovers() -> None:
  mask = np.zeros((40, 40), dtype=np.uint8)
  cv2.circle(mask, (20, 20), 8, 255, -1)
  mask[:20, :] = 0  # lower hemisphere only
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  assert contours
  cx, cy, _r = MiningNodeDetector._disk_center(contours[0], gray=None)
  assert abs(cx - 20.0) < 0.6
  assert abs(cy - 20.0) < 0.6  # enclosing recovers true center


def test_brightness_refine_pulls_up_from_bottom_fringe() -> None:
  """White core + darker bottom crescent → moments low; disk center near core."""
  frame = np.zeros((60, 60, 3), dtype=np.uint8)
  frame[:] = (28, 32, 30)
  true = (30, 28)
  cv2.circle(frame, true, 5, (245, 245, 245), -1, cv2.LINE_AA)
  # bottom shadow crescent (mid-gray) — pulls binary moments down
  for a in range(10, 170):
    rad = math.radians(a)
    x = int(true[0] + 5.4 * math.cos(rad))
    y = int(true[1] + 5.4 * math.sin(rad))
    if y >= true[1] and 0 <= x < 60 and 0 <= y < 60:
      frame[y, x] = (165, 165, 168)

  det = _detector()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask = det._tier_mask(hsv, "gray")
  nodes = det._find_nodes_in_mask(
    mask,
    tier="gray",
    player_x=30,
    player_y=55,
    center_exclusion_radius_px=2.0,
    use_near_center_strict=False,
    gray=gray,
  )
  assert nodes, "expected a gray node"
  node = min(nodes, key=lambda n: math.hypot(n.x - true[0], n.y - true[1]))
  assert abs(node.x - true[0]) < 1.5
  assert abs(node.y - true[1]) < 1.5


def test_road_ribbon_rejected_bright_disk_kept() -> None:
  """Mid-gray elongated pavement must not become a node; white disk must."""
  frame = np.zeros((120, 120, 3), dtype=np.uint8)
  frame[:] = (22, 24, 22)
  # Road ribbon: light-gray pavement in the old false-positive band (~160–190).
  cv2.rectangle(frame, (20, 70), (100, 82), (175, 175, 178), -1)
  # Real mining blip: small bright disk.
  true = (55, 35)
  cv2.circle(frame, true, 4, (240, 240, 242), -1, cv2.LINE_AA)

  det = _detector(center_exclusion_radius_px=8.0, player_center_ratio=(0.5, 0.85))
  scan = det.scan_blips(
    frame,
    player_x=60.0,
    player_y=100.0,
    min_distance_px=10.0,
  )
  assert scan.nodes, "expected the bright disk node"
  for node in scan.nodes:
    assert abs(node.y - 76) > 6.0, f"road locked as node: {node}"
    assert node.circularity >= 0.55
    assert node.area <= 70
  nearest = min(scan.nodes, key=lambda n: math.hypot(n.x - true[0], n.y - true[1]))
  assert abs(nearest.x - true[0]) < 3.0
  assert abs(nearest.y - true[1]) < 3.0


def test_elongated_blob_fails_aspect_filter() -> None:
  mask = np.zeros((40, 80), dtype=np.uint8)
  cv2.ellipse(mask, (40, 20), (28, 6), 0, 0, 360, 255, -1)
  det = _detector(min_circularity=0.2, max_blob_area=400)
  nodes = det._find_nodes_in_mask(
    mask,
    tier="gray",
    player_x=5,
    player_y=20,
    center_exclusion_radius_px=1.0,
    use_near_center_strict=False,
  )
  assert not nodes, f"elongated road-like blob passed: {nodes}"


def test_screenshot_lock_tip_near_visual_center() -> None:
  shot = Path(
    r"C:\Users\daniel\AppData\Roaming\Cursor\User\workspaceStorage"
    r"\empty-window\images\image-8f435cc8-4cc8-40fc-9cdd-1535e7eaf7e2.png"
  )
  if not shot.is_file():
    return  # optional fixture
  img = cv2.imread(str(shot))
  assert img is not None
  det = _detector()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = det._tier_mask(hsv, "gray")
  kernel = np.ones((2, 2), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  nodes = det._find_nodes_in_mask(
    mask,
    tier="gray",
    player_x=5,
    player_y=40,
    center_exclusion_radius_px=1.0,
    use_near_center_strict=False,
    gray=gray,
  )
  # Green tip in this crop was ~bottom of leftmost target; visual center ~ (16,21).
  target = min(nodes, key=lambda n: abs(n.x - 16) + abs(n.y - 21))
  assert abs(target.x - 16.0) < 2.0
  assert abs(target.y - 21.0) < 1.5, f"Y still biased: {target.y}"


if __name__ == "__main__":
  test_half_disk_moments_bias_enclosing_recovers()
  test_brightness_refine_pulls_up_from_bottom_fringe()
  test_road_ribbon_rejected_bright_disk_kept()
  test_elongated_blob_fails_aspect_filter()
  test_screenshot_lock_tip_near_visual_center()
  print("ok")
