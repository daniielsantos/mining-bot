"""Monta builders de percepção a partir do config legado."""

from __future__ import annotations

from typing import Any

from bot import build_arrow_tracker, build_node_detector, build_screen_ui
from minimap_tracker import MinimapArrowTracker
from node_detector import MiningNodeDetector
from screen_ui import MiningScreenUI


def build_perception_stack(cfg: dict[str, Any]) -> tuple[
  MiningNodeDetector,
  MinimapArrowTracker,
  MiningScreenUI,
]:
  return (
    build_node_detector(cfg),
    build_arrow_tracker(cfg),
    build_screen_ui(cfg),
  )
