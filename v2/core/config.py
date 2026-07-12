"""Carrega config: base legada + overlay v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_V2_DIR = Path(__file__).resolve().parent.parent
_MINING_BOT = _V2_DIR.parent
_DEFAULT_V1 = _MINING_BOT / "config.json"
_V2_CONFIG = _V2_DIR / "config.json"
_V2_EXAMPLE = _V2_DIR / "config.example.json"


_LIST_MERGE_KEYS = frozenset({"game_foreground_keywords", "game_process_names"})


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
  out = dict(base)
  for key, value in overlay.items():
    if key in out and isinstance(out[key], dict) and isinstance(value, dict):
      out[key] = _deep_merge(out[key], value)
    elif key in _LIST_MERGE_KEYS and isinstance(out.get(key), list) and isinstance(value, list):
      seen = {str(item).lower() for item in out[key]}
      merged = list(out[key])
      for item in value:
        token = str(item).lower()
        if token not in seen:
          merged.append(item)
          seen.add(token)
      out[key] = merged
    else:
      out[key] = value
  return out


def load_config(path: Path | None = None) -> dict[str, Any]:
  if _DEFAULT_V1.is_file():
    cfg = json.loads(_DEFAULT_V1.read_text(encoding="utf-8"))
  else:
    cfg = {}

  overlay_path = path
  if overlay_path is None:
    overlay_path = _V2_CONFIG if _V2_CONFIG.is_file() else _V2_EXAMPLE

  if overlay_path.is_file():
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    cfg = _deep_merge(cfg, overlay)

  return cfg


def get_minimap_roi(cfg: dict[str, Any]) -> dict[str, int]:
  return dict(cfg["minimap"]["roi"])


def get_hud_roi(cfg: dict[str, Any]) -> dict[str, int]:
  """ROI da barra Mining ore.

  Prefere ``navigation.mining_ore.roi`` (faixa fina centro/baixo — label + barra).
  Fallback: ``screen_roi`` legado (antes muito alto — falhava em 1440p / overlay gordo).
  """
  nav = cfg.get("navigation", {})
  ore = nav.get("mining_ore", cfg.get("mining_ore", {}))
  if isinstance(ore, dict) and isinstance(ore.get("roi"), dict):
    roi = ore["roi"]
    return {
      "left": int(roi["left"]),
      "top": int(roi["top"]),
      "width": int(roi["width"]),
      "height": int(roi["height"]),
    }
  return dict(cfg["screen_roi"])


def get_interaction_roi(cfg: dict[str, Any]) -> dict[str, int]:
  """ROI centro-tela para o prompt E INTERACTION (nao minimapa / barra mining)."""
  nav = cfg.get("navigation", {})
  inter = nav.get("interaction", cfg.get("interaction", {}))
  if isinstance(inter.get("roi"), dict):
    return {
      "left": int(inter["roi"]["left"]),
      "top": int(inter["roi"]["top"]),
      "width": int(inter["roi"]["width"]),
      "height": int(inter["roi"]["height"]),
    }
  res = cfg.get("resolution", {})
  width = int(res.get("width", 2560))
  height = int(res.get("height", 1440))
  # Centro horizontal; um pouco acima do meio vertical (prompt GTA).
  rw = int(inter.get("roi_width", max(520, int(width * 0.32))))
  rh = int(inter.get("roi_height", max(160, int(height * 0.20))))
  left = int(inter.get("roi_left", (width - rw) // 2))
  top = int(inter.get("roi_top", int(height * 0.38) - rh // 2))
  return {"left": left, "top": max(0, top), "width": rw, "height": rh}


def save_overlay_patch(patch: dict[str, Any]) -> Path:
  """Grava merge parcial em v2/config.json (cria se nao existir)."""
  if _V2_CONFIG.is_file():
    existing = json.loads(_V2_CONFIG.read_text(encoding="utf-8"))
  else:
    existing = {}
  merged = _deep_merge(existing, patch)
  _V2_CONFIG.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
  return _V2_CONFIG
