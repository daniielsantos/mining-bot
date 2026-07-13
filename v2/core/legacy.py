"""Monta builders de percepção."""

from __future__ import annotations

from typing import Any

from v2.vendor.builders import (
  build_arrow_tracker,
  build_node_detector,
  build_screen_ui,
)
from v2.vendor.minimap_tracker import MinimapArrowTracker
from v2.vendor.node_detector import MiningNodeDetector
from v2.vendor.screen_ui import MiningScreenUI


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
