"""Testes do detector INTERACTION (E-in-circle primario)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from v2.perception.interaction_detector import InteractionDetector

_ASSETS = Path(
  r"C:\Users\daniel\.cursor\projects\c-Users-daniel-gta-driving-bot\assets"
)
_ASPHALT = (
  _ASSETS
  / "c__Users_daniel_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-72838a8e-1d61-47ef-aee4-65cd51367b95.png"
)
_FAINT = (
  _ASSETS
  / "c__Users_daniel_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-68d84d81-11c9-4cc7-8804-608618ef8c6a.png"
)
_FAR = (
  _ASSETS
  / "c__Users_daniel_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-e1a5a51d-3ab6-496f-b8f3-fddca81705ce.png"
)
_V2_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_E_ASP = _V2_ASSETS / "interaction_e_circle.png"
_E_SAND = _V2_ASSETS / "interaction_e_circle_sand.png"

_LIVE_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5]
_THRESH = 0.72


def _cfg(**overrides):
  base = {
    "navigation": {
      "interaction": {
        "match_threshold": _THRESH,
        "confirm_frames": 1,
        "center_max_dx_frac": 0.28,
        "center_max_dy_frac": 0.35,
        "template": str(_E_ASP),
        "template_sand": str(_E_SAND),
        "scales": list(_LIVE_SCALES),
        "require_text": False,
      }
    }
  }
  base["navigation"]["interaction"].update(overrides)
  return base


def _center_crop(frame):
  h, w = frame.shape[:2]
  rw, rh = int(w * 0.55), int(h * 0.35)
  cx, cy = w // 2, int(h * 0.45)
  x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
  return frame[y1 : y1 + rh, x1 : x1 + rw]


def _pad_prompt(frame, h: int = 280, w: int = 800) -> np.ndarray:
  mean = frame.mean(axis=(0, 1)).astype(np.uint8)
  canvas = np.zeros((h, w, 3), dtype=np.uint8)
  canvas[:] = mean
  th, tw = frame.shape[:2]
  y, x = (h - th) // 2, (w - tw) // 2
  canvas[y : y + th, x : x + tw] = frame
  return canvas


def _synth_sand(h: int = 280, w: int = 800) -> np.ndarray:
  rng = np.random.default_rng(0)
  base = np.full((h, w, 3), 220, dtype=np.uint8)
  noise = rng.integers(-12, 12, size=base.shape, dtype=np.int16)
  return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _synth_dark(h: int = 280, w: int = 800) -> np.ndarray:
  rng = np.random.default_rng(1)
  base = np.full((h, w, 3), 70, dtype=np.uint8)
  noise = rng.integers(-20, 20, size=base.shape, dtype=np.int16)
  return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _raw_gates_pass(hit, threshold: float = _THRESH) -> bool:
  return (
    hit.score >= threshold
    and hit.contrast_ok
    and hit.center_ok
    and hit.e_circle_ok
  )


@pytest.mark.skipif(not _E_ASP.is_file(), reason="E asphalt template missing")
def test_synth_no_prompt_never_ready():
  det = InteractionDetector(_cfg(confirm_frames=1))
  for bg in (_synth_sand(), _synth_dark()):
    hit = det.detect(bg)
    assert hit.found is False
    assert not _raw_gates_pass(hit)


@pytest.mark.skipif(not _E_ASP.is_file(), reason="E asphalt template missing")
def test_asphalt_with_e_hits():
  if not _ASPHALT.is_file():
    pytest.skip("asphalt screenshot missing")
  det = InteractionDetector(_cfg())
  frame = cv2.imread(str(_ASPHALT))
  assert frame is not None
  hit = det.detect(_pad_prompt(frame))
  assert hit.score >= _THRESH
  assert hit.contrast_ok is True
  assert hit.center_ok is True
  assert hit.e_circle_ok is True
  assert hit.found is True


@pytest.mark.skipif(not _E_SAND.is_file(), reason="E sand template missing")
def test_sand_faint_with_e_hits():
  if not _FAINT.is_file():
    pytest.skip("faint screenshot missing")
  det = InteractionDetector(_cfg())
  frame = cv2.imread(str(_FAINT))
  assert frame is not None
  hit = det.detect(_pad_prompt(frame))
  assert hit.score >= _THRESH
  assert hit.contrast_ok is True
  assert hit.center_ok is True
  assert hit.e_circle_ok is True
  assert hit.found is True


@pytest.mark.skipif(not _E_ASP.is_file(), reason="E asphalt template missing")
def test_far_no_prompt_not_found():
  if not _FAR.is_file():
    pytest.skip("far screenshot missing")
  det = InteractionDetector(_cfg(match_threshold=_THRESH, confirm_frames=1))
  frame = cv2.imread(str(_FAR))
  assert frame is not None
  hit = det.detect(_center_crop(frame))
  assert hit.found is False
  assert not _raw_gates_pass(hit)


@pytest.mark.skipif(not _E_ASP.is_file(), reason="E asphalt template missing")
def test_confirm_frames_require_streak():
  if not _ASPHALT.is_file():
    pytest.skip("asphalt screenshot missing")
  det = InteractionDetector(_cfg(confirm_frames=3))
  frame = cv2.imread(str(_ASPHALT))
  assert frame is not None
  crop = _pad_prompt(frame)
  assert det.detect(crop).found is False
  assert det.detect(crop).found is False
  assert det.detect(crop).found is True


@pytest.mark.skipif(not _E_ASP.is_file(), reason="E asphalt template missing")
def test_random_white_letter_rejected():
  """Letra branca sem circulo nao deve passar no gate e_circle."""
  canvas = _synth_dark()
  cv2.putText(
    canvas,
    "E",
    (canvas.shape[1] // 2 - 20, canvas.shape[0] // 2 + 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    2.0,
    (255, 255, 255),
    3,
    cv2.LINE_AA,
  )
  det = InteractionDetector(_cfg(confirm_frames=1))
  hit = det.detect(canvas)
  assert hit.found is False
  assert hit.e_circle_ok is False or hit.score < _THRESH
