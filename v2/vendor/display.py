from __future__ import annotations

import cv2
import numpy as np


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((40, width, 3), dtype=np.uint8)
    scale = width / max(image.shape[1], 1)
    height = max(1, int(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_status_bar(panel: np.ndarray, lines: list[str], *, bar_height: int = 48) -> np.ndarray:
    bar = np.zeros((bar_height, panel.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(
            bar,
            line,
            (8, 18 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([panel, bar])


def compose_preview(
    minimap_dbg: np.ndarray,
    progress_crop: np.ndarray,
    *,
    panel_width: int = 380,
) -> np.ndarray:
    """Minimapa em cima, barra de progresso embaixo — mesma largura."""
    mini = fit_width(minimap_dbg, panel_width)
    prog = fit_width(progress_crop, panel_width)
    if prog.shape[0] < 36:
        prog = cv2.resize(prog, (panel_width, 36), interpolation=cv2.INTER_AREA)
    return np.vstack([mini, prog])
