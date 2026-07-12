"""Calibra HSV do nó cinza e grava em v2/config.json.

Uso no preview (python -m v2.main --preview):
  G  — arma o próximo clique no minimapa
  clique esquerdo no disco branco do nó — amostra HSV, salva, reaplica no detector

CLI dedicado:
  cd mining_bot
  python -m v2.calibrate_gray
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from v2.core.config import save_overlay_patch


def clamp(value: int, lo: int, hi: int) -> int:
  return max(lo, min(hi, value))


def pick_gray_node_hsv(
  frame_bgr: np.ndarray,
  x: int,
  y: int,
  *,
  radius: int = 2,
  s_tol: int = 40,
  v_tol: int = 28,
) -> tuple[list[int], list[int], tuple[int, int, int]]:
  """Amostra o núcleo do blip cinza/branco com tolerância moderada.

  Estradas mid-gray (~V 150–170) ficam fora quando o clique é no disco claro.
  H fica 0–179 (acromático); S/V seguem o pixel amostrado ± tolerância.
  S_tol apertado (~28) fazia flicker → ghost abandon → lock thrash.
  """
  hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
  h, w = hsv.shape[:2]
  x = clamp(int(x), 0, w - 1)
  y = clamp(int(y), 0, h - 1)
  x0, x1 = max(0, x - radius), min(w, x + radius + 1)
  y0, y1 = max(0, y - radius), min(h, y + radius + 1)
  patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
  # Núcleo do nó = pixel mais claro da vizinhança (evita franja AA).
  idx = int(np.argmax(patch[:, 2]))
  h_s, s_s, v_s = (int(patch[idx, 0]), int(patch[idx, 1]), int(patch[idx, 2]))

  lower = [0, 0, clamp(v_s - v_tol, 0, 255)]
  upper = [179, clamp(s_s + s_tol, 0, 255), 255]
  return lower, upper, (h_s, s_s, v_s)


def pick_tier_hsv(
  frame_bgr: np.ndarray,
  x: int,
  y: int,
  tier: str = "gray",
) -> tuple[list[int], list[int], tuple[int, int, int]]:
  """HSV para um tier; gray usa caixa estreita, coloridos usam janela em H."""
  if tier.lower() == "gray":
    return pick_gray_node_hsv(frame_bgr, x, y)

  hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
  h, w = hsv.shape[:2]
  x = clamp(int(x), 0, w - 1)
  y = clamp(int(y), 0, h - 1)
  hh, ss, vv = [int(c) for c in hsv[y, x]]
  lower = [clamp(hh - 10, 0, 179), clamp(ss - 50, 0, 255), clamp(vv - 50, 0, 255)]
  upper = [clamp(hh + 10, 0, 179), 255, 255]
  return lower, upper, (hh, ss, vv)


def gray_config_patch(lower: list[int], upper: list[int]) -> dict[str, Any]:
  """Patch para save_overlay_patch — bounds = fonte da verdade (sem expand)."""
  return {
    "tier_colors_hsv": {
      "gray": {"lower": list(lower), "upper": list(upper)},
    },
    "gray_achromatic_expand": False,
    "gray_achromatic_v_min": int(lower[2]),
    "gray_achromatic_s_max": int(upper[1]),
  }


def save_gray_color(lower: list[int], upper: list[int]):
  """Persiste cor gray em mining_bot/v2/config.json."""
  return save_overlay_patch(gray_config_patch(lower, upper))


def apply_gray_to_runtime(
  cfg: dict[str, Any],
  detector: Any,
  lower: list[int],
  upper: list[int],
) -> None:
  """Atualiza cfg + detector em memória (sem reiniciar o bot)."""
  colors = cfg.setdefault("tier_colors_hsv", {})
  colors["gray"] = {"lower": list(lower), "upper": list(upper)}
  cfg["gray_achromatic_expand"] = False
  cfg["gray_achromatic_v_min"] = int(lower[2])
  cfg["gray_achromatic_s_max"] = int(upper[1])
  detector.tier_colors_hsv = dict(cfg["tier_colors_hsv"])
  if hasattr(detector, "gray_achromatic_expand"):
    detector.gray_achromatic_expand = False
  if hasattr(detector, "gray_achromatic_v_min"):
    detector.gray_achromatic_v_min = int(lower[2])
  if hasattr(detector, "gray_achromatic_s_max"):
    detector.gray_achromatic_s_max = int(upper[1])


def map_preview_click_to_minimap(
  click_x: int,
  click_y: int,
  *,
  minimap_hw: tuple[int, int],
  panel_width: int,
) -> tuple[int, int] | None:
  """Mapeia clique na janela Mining v2 → pixel do minimapa original."""
  mh, mw = int(minimap_hw[0]), int(minimap_hw[1])
  if mw <= 0 or mh <= 0 or panel_width <= 0:
    return None
  scale = panel_width / float(mw)
  mini_h = max(1, int(mh * scale))
  if click_y < 0 or click_y >= mini_h or click_x < 0 or click_x >= panel_width:
    return None
  mx = clamp(int(round(click_x / scale)), 0, mw - 1)
  my = clamp(int(round(click_y / scale)), 0, mh - 1)
  return mx, my
