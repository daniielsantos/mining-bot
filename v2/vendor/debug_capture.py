from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _safe_token(value: str, max_len: int = 28) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return cleaned[:max_len] or "evt"


class SessionFrameRecorder:
    """Salva frames do overlay por sessao (F6 ligado) para debug offline."""

    def __init__(
        self,
        base_dir: Path,
        *,
        interval_s: float = 0.5,
        max_frames: int = 200,
        enabled: bool = True,
        save_minimap: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.interval_s = interval_s
        self.max_frames = max_frames
        self.enabled = enabled
        self.save_minimap = save_minimap
        self._session_dir: Path | None = None
        self._frame_idx = 0
        self._last_save_at = 0.0
        self._last_action = ""
        self._last_state = ""

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start_session(self) -> Path | None:
        if not self.enabled:
            return None
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = self.base_dir / f"sessao_{stamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        self._last_save_at = 0.0
        self._last_action = ""
        self._last_state = ""
        readme = self._session_dir / "README.txt"
        readme.write_text(
            "Frames de debug do mining bot.\n\n"
            "Arquivos:\n"
            "  NNNN_estado-acao.jpg       — overlay (tile + status)\n"
            "  NNNN_estado-acao.json      — metadados\n"
            "  NNNN_estado-acao_minimap.jpg — minimap cru (se save_minimap=true)\n\n"
            "Campos uteis no JSON (simple_bot / bot):\n"
            "  player_tile_x/y     — posicao no mapa tile\n"
            "  correlate_shift_mag — shift da correlacao (mapa rolando)\n"
            "  frame_delta_mag     — movimento frame-a-frame\n"
            "  facing_delta        — rotacao da seta entre frames\n"
            "  pure_turn           — true = posicao congelada (so girando)\n"
            "  tile_dist_px        — distancia no tile (deve cair ao andar)\n"
            "  screen_dist_px      — distancia seta→alvo na tela\n"
            "  tile_bearing_deg    — rumo estavel no tile\n"
            "  heading_error_deg   — rumo seta→alvo na tela\n"
            "  blip_snapped        — alvo colado no blip visivel\n"
            "  stuck_frames        — contador de distancia presa\n",
            encoding="utf-8",
        )
        return self._session_dir

    def end_session(self) -> None:
        self._session_dir = None
        self._frame_idx = 0

    def maybe_save(
        self,
        image_bgr: np.ndarray,
        meta: dict,
        *,
        force: bool = False,
        minimap_bgr: np.ndarray | None = None,
    ) -> str | None:
        if not self.enabled or self._session_dir is None or image_bgr.size == 0:
            return None
        if self._frame_idx >= self.max_frames:
            return None

        now = time.perf_counter()
        action = str(meta.get("action", ""))
        state = str(meta.get("state", ""))
        should_save = (
            force
            or action != self._last_action
            or state != self._last_state
            or (now - self._last_save_at) >= self.interval_s
        )
        if not should_save:
            return None

        self._frame_idx += 1
        self._last_save_at = now
        self._last_action = action
        self._last_state = state

        token = _safe_token(f"{state}-{action}")
        stem = f"{self._frame_idx:04d}_{token}"
        image_path = self._session_dir / f"{stem}.jpg"
        meta_path = self._session_dir / f"{stem}.json"

        payload = {
            **meta,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": self._frame_idx,
        }
        cv2.imwrite(str(image_path), image_bgr)
        if (
            self.save_minimap
            and minimap_bgr is not None
            and minimap_bgr.size > 0
        ):
            cv2.imwrite(str(self._session_dir / f"{stem}_minimap.jpg"), minimap_bgr)
            payload["minimap_file"] = f"{stem}_minimap.jpg"
        meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(image_path)
