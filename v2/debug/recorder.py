"""Gravação de sessão debug v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from v2.vendor.debug_capture import SessionFrameRecorder
from v2.core.types import FrameContext


class SessionRecorder:
  def __init__(self, cfg: dict[str, Any], base_dir: Path) -> None:
    dbg = cfg.get("debug", {})
    self._recorder = SessionFrameRecorder(
      base_dir,
      interval_s=float(dbg.get("record_interval_s", 0.25)),
      max_frames=int(dbg.get("max_frames", 400)),
      enabled=bool(dbg.get("record_enabled", True)),
      save_minimap=bool(dbg.get("save_minimap", True)),
    )

  @property
  def session_dir(self) -> Path | None:
    return self._recorder.session_dir

  def start(self, reason: str) -> Path | None:
    return self._recorder.start_session()

  def maybe_save(
    self,
    image: np.ndarray,
    ctx: FrameContext,
    *,
    enabled: bool,
    game_focus: bool,
    force: bool = False,
  ) -> None:
    meta = ctx.debug_dict()
    meta["enabled"] = enabled
    meta["game_focus"] = game_focus
    self._recorder.maybe_save(
      image,
      meta,
      force=force,
      minimap_bgr=ctx.minimap_bgr if self._recorder.save_minimap else None,
    )

  def stop(self) -> None:
    self._recorder.end_session()
