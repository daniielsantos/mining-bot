"""Offline tests: MiningOreDetector vs template / negative crops."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from v2.core.config import get_hud_roi, load_config
from v2.perception.mining_ore_detector import MiningOreDetector

_V2 = Path(__file__).resolve().parents[1]
_TMPL = _V2 / "assets" / "mining_ore_label.png"
_ASSETS = Path(
  r"C:\Users\daniel\.cursor\projects\c-Users-daniel-gta-driving-bot\assets"
)
# Mining ore at 0% — tight dark bar source for template + positive fixture.
_CLOSEUP = (
  _ASSETS
  / "c__Users_daniel_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-2876d23a-d9bc-4d2f-a1a2-cb9436603933.png"
)
_OVERLAY = (
  _ASSETS
  / "c__Users_daniel_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-0cbaddbe-62ec-4e9d-8120-9025ecac0b6f.png"
)
_EMPTY_OVERLAY = (
  _ASSETS
  / "c__Users_daniel_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-8d3d3262-ee1b-4866-a34e-d6233e89c042.png"
)

_THRESH = 0.70
_EMPTY_MAX = 0.40


def _cfg(**overrides):
  base = {
    "navigation": {
      "mining_ore": {
        "match_threshold": _THRESH,
        "confirm_frames": 1,
        "require_ui_contrast": True,
        "contrast_bright_min": 100,
        "contrast_gap_min": 80,
        "contrast_check_floor": 0.30,
        "contrast_fail_scale": 0.50,
        "contrast_fail_cap": 0.35,
        "hold_min": 0.70,
        "template": str(_TMPL),
        "scales": [
          0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0
        ],
      }
    }
  }
  base["navigation"]["mining_ore"].update(overrides)
  return base


def _native_with_bar() -> np.ndarray:
  """Simula ROI larga com a barra Mining ore (centro/baixo)."""
  bar = cv2.imread(str(_CLOSEUP))
  assert bar is not None
  canvas = np.full((480, 1000, 3), 70, dtype=np.uint8)
  rng = np.random.default_rng(3)
  canvas = np.clip(
    canvas.astype(np.int16) + rng.integers(-20, 30, canvas.shape), 0, 255
  ).astype(np.uint8)
  bh, bw = bar.shape[:2]
  y0 = max(0, (480 - bh) // 2)
  x0 = max(0, (1000 - bw) // 2)
  canvas[y0 : y0 + bh, x0 : x0 + bw] = bar
  return canvas


def _native_without() -> np.ndarray:
  rng = np.random.default_rng(4)
  return np.clip(
    np.full((480, 1000, 3), 70, dtype=np.int16)
    + rng.integers(-25, 40, (480, 1000, 3)),
    0,
    255,
  ).astype(np.uint8)


@pytest.mark.skipif(not _TMPL.is_file(), reason="mining_ore_label.png missing")
def test_native_with_mining_ore_hits():
  if not _CLOSEUP.is_file():
    pytest.skip("closeup screenshot missing")
  det = MiningOreDetector(_cfg())
  hit = det.detect(_native_with_bar())
  assert hit.score >= _THRESH
  assert hit.found is True or hit.raw_hit is True


@pytest.mark.skipif(not _TMPL.is_file(), reason="mining_ore_label.png missing")
def test_native_without_mining_ore_misses():
  det = MiningOreDetector(_cfg())
  hit = det.detect(_native_without())
  assert hit.found is False
  assert hit.score < _EMPTY_MAX


@pytest.mark.skipif(not _TMPL.is_file(), reason="mining_ore_label.png missing")
def test_empty_hud_overlay_scores_low():
  """User screenshot: middle HUD empty of Mining ore — must be <0.40."""
  if not _EMPTY_OVERLAY.is_file():
    pytest.skip("empty overlay screenshot missing")
  det = MiningOreDetector(_cfg())
  overlay = cv2.imread(str(_EMPTY_OVERLAY))
  assert overlay is not None
  # Preview middle strip (same band as overlay HUD panel).
  for y0, y1 in ((200, 280), (210, 300), (180, 260)):
    hit = det.detect(overlay[y0:y1, :])
    assert hit.found is False
    assert hit.raw_hit is False
    assert hit.score < _EMPTY_MAX, f"band {y0}:{y1} score={hit.score:.3f}"


@pytest.mark.skipif(not _TMPL.is_file(), reason="mining_ore_label.png missing")
def test_closeup_direct_hits():
  if not _CLOSEUP.is_file():
    pytest.skip("closeup screenshot missing")
  det = MiningOreDetector(_cfg())
  frame = cv2.imread(str(_CLOSEUP))
  assert frame is not None
  hit = det.detect(frame)
  assert hit.score >= _THRESH
  assert hit.found is True or hit.raw_hit is True


@pytest.mark.skipif(not _TMPL.is_file(), reason="mining_ore_label.png missing")
def test_overlay_hud_band_hits():
  """Preview composite: faixa do HUD com Mining ore visivel (downscale)."""
  if not _OVERLAY.is_file():
    pytest.skip("overlay screenshot missing")
  # Preview e downscale — contrast gate nativo nao se aplica; so score.
  det = MiningOreDetector(
    _cfg(confirm_frames=1, match_threshold=0.55, require_ui_contrast=False)
  )
  overlay = cv2.imread(str(_OVERLAY))
  assert overlay is not None
  hud = overlay[210:300, :]
  hit = det.detect(hud)
  assert hit.score >= 0.55
  assert hit.found is True


@pytest.mark.skipif(not _TMPL.is_file(), reason="mining_ore_label.png missing")
def test_confirm_frames_streak():
  if not _CLOSEUP.is_file():
    pytest.skip("closeup screenshot missing")
  det = MiningOreDetector(_cfg(confirm_frames=3))
  crop = _native_with_bar()
  assert det.detect(crop).found is False
  assert det.detect(crop).found is False
  assert det.detect(crop).found is True


def test_hud_roi_prefers_mining_ore_roi():
  cfg = load_config()
  roi = get_hud_roi(cfg)
  # Faixa fina da barra (nao strip 1000x100+ com fundo).
  assert 24 <= roi["height"] <= 120
  assert roi["width"] >= 200
  assert roi["width"] <= 900
  assert "left" in roi and "top" in roi
