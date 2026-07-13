from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_REFERENCE_ARROW_PATH = _PACKAGE_DIR / "assets" / "player_arrow_template.png"


@dataclass
class ArrowResult:
    player_tracked: bool
    arrow_detected: bool
    player_x: float
    player_y: float
    arrow_tip_x: float | None = None
    arrow_tip_y: float | None = None
    arrow_angle_deg: float | None = None
    anchor_x: float | None = None
    anchor_y: float | None = None

    def pivot(self) -> tuple[float, float]:
        """Posicao real da seta no minimapa (centroide do icone)."""
        if self.anchor_x is not None and self.anchor_y is not None:
            return self.anchor_x, self.anchor_y
        return self.player_x, self.player_y


class MinimapArrowTracker:
    """Detecta a seta do jogador no minimapa (a pe ou veiculo)."""

    def __init__(
        self,
        *,
        player_center_ratio: tuple[float, float] = (0.5, 0.5),
        arrow_gray_min: int = 145,
        arrow_gray_max: int = 175,
        arrow_white_min: int = 165,
        arrow_min_area: int = 8,
        arrow_max_area: int = 220,
        player_position_smoothing: float = 0.62,
        arrow_search_radius_px: float = 42.0,
        arrow_min_tip_dist_px: float = 5.0,
        arrow_max_tip_dist_px: float = 30.0,
        arrow_max_centroid_dist_px: float = 22.0,
        arrow_max_tip_jump_px: float = 12.0,
        arrow_max_lost_frames: int = 36,
        fixed_player_anchor: bool = True,
        player_center_calibrated: bool = False,
    ) -> None:
        self.player_center_ratio = player_center_ratio
        self.fixed_player_anchor = fixed_player_anchor
        self.player_center_calibrated = player_center_calibrated
        self.arrow_gray_min = arrow_gray_min
        self.arrow_gray_max = arrow_gray_max
        self.arrow_white_min = arrow_white_min
        self.arrow_min_area = arrow_min_area
        self.arrow_max_area = arrow_max_area
        self.player_position_smoothing = player_position_smoothing
        self.arrow_search_radius_px = arrow_search_radius_px
        self.arrow_min_tip_dist_px = arrow_min_tip_dist_px
        self.arrow_max_tip_dist_px = arrow_max_tip_dist_px
        self.arrow_max_centroid_dist_px = arrow_max_centroid_dist_px
        self.arrow_max_tip_jump_px = arrow_max_tip_jump_px
        self._smooth_tip_x: float | None = None
        self._smooth_tip_y: float | None = None
        self._last_facing_deg: float | None = None
        self._player_lost_frames = 0
        self._player_max_lost_frames = max(int(arrow_max_lost_frames), 8)
        self._last_arrow_contour: np.ndarray | None = None
        self._last_anchor_x: float | None = None
        self._last_anchor_y: float | None = None
        self._arrow_templates: list[tuple[float, np.ndarray, float, float, float, float]] | None = None
        self._reference_templates: list[tuple[float, np.ndarray, float, float, float, float]] | None = None
        self._circle_template: np.ndarray | None = None
        self._node_blob_cache: list[tuple[float, float, float, float]] | None = None
        self._node_blob_frame = -999
        self._detect_frame = 0
        self._auto_anchor_x: float | None = None
        self._auto_anchor_y: float | None = None

    def _create_circle_template(self, size: int = 13) -> np.ndarray:
        t = np.full((size, size), 56, dtype=np.uint8)
        cv2.circle(t, (size // 2, size // 2), max(size // 2 - 2, 2), 140, -1)
        edge = cv2.Canny(t, 45, 130)
        t[edge > 0] = 34
        return t

    def _circle_template_match_at(self, gray: np.ndarray, cx: int, cy: int) -> float:
        if self._circle_template is None:
            self._circle_template = self._create_circle_template()
        tmpl = self._circle_template
        th, tw = tmpl.shape
        x0 = int(round(cx - tw * 0.5))
        y0 = int(round(cy - th * 0.5))
        if x0 < 0 or y0 < 0 or x0 + tw > gray.shape[1] or y0 + th > gray.shape[0]:
            return 0.0
        patch = gray[y0 : y0 + th, x0 : x0 + tw]
        return float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])

    @staticmethod
    def _rotate_point(
        x: float,
        y: float,
        cx: float,
        cy: float,
        angle_deg: float,
    ) -> tuple[float, float]:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = x - cx
        dy = y - cy
        return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a

    def _create_arrow_template(self, size: int = 13) -> tuple[np.ndarray, float, float, float, float]:
        """Seta sintetica: metade clara + metade cinza sobre fundo escuro."""
        t = np.full((size, size), 56, dtype=np.uint8)
        c = size // 2
        pts = np.array(
            [
                [c + 3, c - 4],
                [c + 5, c + 4],
                [c - 4, c + 2],
            ],
            dtype=np.int32,
        )
        poly = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(poly, pts, 255)
        for y in range(size):
            for x in range(size):
                if poly[y, x] == 0:
                    continue
                t[y, x] = 205 if x >= c - 1 else 118
        edge = cv2.Canny(t, 45, 130)
        t[edge > 0] = 34
        # Pivot = centro da base (cauda) — igual ao icone do GTA.
        pivot_x = (pts[1][0] + pts[2][0] + 1) / 2.0
        pivot_y = (pts[1][1] + pts[2][1] + 1) / 2.0
        tip_x = float(pts[0][0] + 0.5)
        tip_y = float(pts[0][1] + 0.5)
        return t, pivot_x, pivot_y, tip_x, tip_y

    def _rotated_arrow_templates(self) -> list[tuple[float, np.ndarray, float, float, float, float]]:
        if self._arrow_templates is not None:
            return self._arrow_templates
        base, pivot_x, pivot_y, tip_x, tip_y = self._create_arrow_template()
        h, w = base.shape
        cx, cy = w / 2.0, h / 2.0
        templates: list[tuple[float, np.ndarray, float, float, float, float]] = []
        for angle in range(0, 360, 15):
            m = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
            rot = cv2.warpAffine(
                base,
                m,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=56,
            )
            rp_x, rp_y = self._rotate_point(pivot_x, pivot_y, cx, cy, float(angle))
            rt_x, rt_y = self._rotate_point(tip_x, tip_y, cx, cy, float(angle))
            templates.append((float(angle), rot, rp_x, rp_y, rt_x, rt_y))
        self._arrow_templates = templates
        return templates

    def _reference_arrow_pivot_tip(
        self, gray: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        Template aponta pra CIMA: nariz = pixels do topo do corpo
        (nao 'branco mais longe' — isso pega a asa da base).
        """
        h, w = gray.shape[:2]
        body = (gray >= 120) & (gray <= 250)
        ys, xs = np.where(body)
        if xs.size < 4:
            return w / 2.0, h / 2.0, w / 2.0, h * 0.18
        pivot_x = float(np.mean(xs))
        pivot_y = float(np.mean(ys))
        # Topo = nariz (template sempre com frente para cima).
        min_y = int(ys.min())
        top = ys <= (min_y + 2)
        tip_x = float(np.mean(xs[top]))
        tip_y = float(np.mean(ys[top]))
        return pivot_x, pivot_y, tip_x, tip_y

    def _reference_rotated_templates(
        self,
    ) -> list[tuple[float, np.ndarray, float, float, float, float]]:
        if self._reference_templates is not None:
            return self._reference_templates
        if not _REFERENCE_ARROW_PATH.is_file():
            self._reference_templates = []
            return self._reference_templates

        raw = cv2.imread(str(_REFERENCE_ARROW_PATH), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            self._reference_templates = []
            return self._reference_templates

        base = cv2.resize(raw, (24, 26), interpolation=cv2.INTER_AREA)
        pivot_x, pivot_y, tip_x, tip_y = self._reference_arrow_pivot_tip(base)
        h, w = base.shape
        cx, cy = w / 2.0, h / 2.0
        templates: list[tuple[float, np.ndarray, float, float, float, float]] = []
        for angle in range(0, 360, 5):
            matrix = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
            rot = cv2.warpAffine(
                base,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=56,
            )
            rp_x, rp_y = self._rotate_point(pivot_x, pivot_y, cx, cy, float(angle))
            rt_x, rt_y = self._rotate_point(tip_x, tip_y, cx, cy, float(angle))
            templates.append((float(angle), rot, rp_x, rp_y, rt_x, rt_y))
        self._reference_templates = templates
        return templates

    def _template_sets(
        self,
    ) -> list[tuple[float, np.ndarray, float, float, float, float]]:
        ref = self._reference_rotated_templates()
        if ref:
            return ref
        return self._rotated_arrow_templates()

    def _template_prior_bonus(self, cx: float, cy: float, width: int, height: int) -> float:
        bonus = 0.0
        if self._last_anchor_x is not None and self._last_anchor_y is not None:
            anchor_dist = math.hypot(cx - self._last_anchor_x, cy - self._last_anchor_y)
            bonus += max(0.0, 0.14 - anchor_dist / 70.0)
        nx = cx / max(width, 1)
        ny = cy / max(height, 1)
        px, py = self.player_center_ratio
        dist = math.hypot(nx - px, ny - py)
        bonus += max(0.0, 0.08 - dist) * 2.5
        expected_y = height * self.player_center_ratio[1]
        bonus += max(0.0, 0.12 - abs(cy - expected_y) / max(height * 0.22, 1.0))
        return bonus

    def _near_last_anchor(self, ax: float, ay: float, *, radius_px: float = 48.0) -> bool:
        if self._last_anchor_x is None or self._last_anchor_y is None:
            return False
        return math.hypot(ax - self._last_anchor_x, ay - self._last_anchor_y) <= radius_px

    def _get_node_blobs(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search_mask: np.ndarray,
    ) -> list[tuple[float, float, float, float]]:
        self._detect_frame += 1
        if (
            self._node_blob_cache is not None
            and self._detect_frame - self._node_blob_frame < 12
        ):
            return self._node_blob_cache
        blobs = self._find_mining_node_blobs(gray, search_mask)
        blobs.extend(self._find_colored_node_blobs(hsv, search_mask))
        self._node_blob_cache = self._dedupe_blobs(blobs)
        self._node_blob_frame = self._detect_frame
        return self._node_blob_cache

    @staticmethod
    def _dedupe_blobs(
        blobs: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        deduped: list[tuple[float, float, float, float]] = []
        for blob in blobs:
            if any(
                math.hypot(blob[0] - kept[0], blob[1] - kept[1]) < max(blob[2], kept[2]) + 4.0
                for kept in deduped
            ):
                continue
            deduped.append(blob)
        return deduped

    def _find_colored_node_blobs(
        self,
        hsv: np.ndarray,
        search_mask: np.ndarray,
    ) -> list[tuple[float, float, float, float]]:
        """Nos coloridos (laranja, ciano, etc.) — nunca sao a seta do jogador."""
        blobs: list[tuple[float, float, float, float]] = []
        color_ranges = [
            (np.array([2, 95, 150], dtype=np.uint8), np.array([26, 255, 255], dtype=np.uint8)),
            (np.array([78, 110, 120], dtype=np.uint8), np.array([108, 255, 255], dtype=np.uint8)),
            (np.array([130, 80, 120], dtype=np.uint8), np.array([165, 255, 255], dtype=np.uint8)),
            # Nos desaturados no minimapa (laranja/ciano escuro — falham nos ranges acima).
            (np.array([0, 38, 42], dtype=np.uint8), np.array([28, 150, 210], dtype=np.uint8)),
            (np.array([82, 38, 42], dtype=np.uint8), np.array([118, 150, 210], dtype=np.uint8)),
        ]
        kernel = np.ones((2, 2), np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_and(mask, search_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 8.0 or area > 320.0:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] < 1.0:
                    continue
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
                radius = max(math.sqrt(area / math.pi), 3.0)
                blobs.append((cx, cy, radius, 0.85))
        return blobs

    def _is_achromatic_at(self, hsv: np.ndarray, cx: int, cy: int, *, half: int = 8) -> bool:
        """Seta do jogador = branco/cinza (baixa saturacao). Nós coloridos falham aqui."""
        if not (half <= cx < hsv.shape[1] - half and half <= cy < hsv.shape[0] - half):
            return True
        patch = hsv[cy - half : cy + half + 1, cx - half : cx + half + 1]
        sat = patch[:, :, 1].astype(np.float32)
        val = patch[:, :, 2].astype(np.float32)
        bright = val >= 105
        if int(np.count_nonzero(bright)) < 4:
            return False
        sats = sat[bright]
        mean_sat = float(sats.mean())
        max_sat = float(sats.max())
        return mean_sat < 72.0 and max_sat < 105.0

    def _patch_has_mining_color(
        self,
        hsv: np.ndarray,
        cx: int,
        cy: int,
        *,
        half: int = 5,
    ) -> bool:
        """True se o miolo do patch tem pixels claramente coloridos (nos de mineracao)."""
        if not (half <= cx < hsv.shape[1] - half and half <= cy < hsv.shape[0] - half):
            return False
        patch = hsv[cy - half : cy + half + 1, cx - half : cx + half + 1]
        sat = patch[:, :, 1].astype(np.float32)
        val = patch[:, :, 2].astype(np.float32)
        colored = (sat >= 50.0) & (val >= 58.0)
        return float(colored.mean()) >= 0.48

    def _has_bicolor_arrow_patch(
        self,
        gray: np.ndarray,
        cx: int,
        cy: int,
        *,
        half: int = 8,
    ) -> bool:
        """Seta real = branco + cinza no mesmo patch; circulos coloridos nao."""
        if not (half <= cx < gray.shape[1] - half and half <= cy < gray.shape[0] - half):
            return True
        patch = gray[cy - half : cy + half + 1, cx - half : cx + half + 1]
        white_frac = float((patch >= 150).mean())
        gray_frac = float(((patch >= 70) & (patch <= 145)).mean())
        return white_frac >= 0.04 and gray_frac >= 0.04

    def _iter_template_peaks(
        self,
        res: np.ndarray,
        *,
        min_match: float,
        max_peaks: int = 10,
        suppress: int = 7,
    ):
        work = res.copy()
        th, tw = res.shape
        pad_y = max(suppress, 1)
        pad_x = max(suppress, 1)
        for _ in range(max_peaks):
            _min, max_val, _min_loc, max_loc = cv2.minMaxLoc(work)
            if max_val < min_match:
                break
            yield float(max_val), max_loc
            px, py = max_loc
            y0 = max(0, py - pad_y)
            y1 = min(th, py + pad_y + 1)
            x0 = max(0, px - pad_x)
            x1 = min(tw, px + pad_x + 1)
            work[y0:y1, x0:x1] = -1.0

    def _score_template_anchor(
        self,
        *,
        match_val: float,
        ax: float,
        ay: float,
        width: int,
        height: int,
    ) -> float:
        score = match_val + self._template_prior_bonus(ax, ay, width, height)
        if self._last_anchor_x is not None and self._last_anchor_y is not None:
            score += max(
                0.0,
                0.08 - math.hypot(ax - self._last_anchor_x, ay - self._last_anchor_y) / 100.0,
            )
        return score

    def _accept_template_anchor(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
        *,
        ax: float,
        ay: float,
        probe_x: float,
        probe_y: float,
        match_val: float,
        width: int,
        height: int,
        relax: bool = False,
    ) -> float | None:
        ix, iy = int(round(probe_x)), int(round(probe_y))
        if not (0 <= ix < width and 0 <= iy < height and search[iy, ix] > 0):
            return None
        if (
            not self.fixed_player_anchor
            and self._last_anchor_x is None
            and ay < height * 0.52
        ):
            return None
        if self._point_on_mining_node(ax, ay, node_blobs, min_node_radius=4.0):
            return None
        if self._reject_circle_node_at(gray, ix, iy):
            return None
        if (
            self._last_anchor_x is not None
            and self._last_anchor_y is not None
            and math.hypot(ax - self._last_anchor_x, ay - self._last_anchor_y) > 36.0
            and match_val < 0.52
        ):
            return None
        tracking = relax or self._near_last_anchor(ax, ay)
        if not tracking and self._patch_has_mining_color(hsv, ix, iy):
            return None
        if tracking and self._patch_has_mining_color(hsv, ix, iy) and match_val < 0.44:
            return None
        if not self._is_achromatic_at(hsv, ix, iy):
            return None
        if not tracking and not self._has_bicolor_arrow_patch(gray, ix, iy):
            return None
        if tracking and not self._has_bicolor_arrow_patch(gray, ix, iy) and match_val < 0.34:
            return None
        if not self._arrow_beats_circle_vals(match_val, gray, ix, iy):
            return None
        half = 8
        if half <= ix < gray.shape[1] - half and half <= iy < gray.shape[0] - half:
            patch = gray[iy - half : iy + half + 1, ix - half : ix + half + 1]
            metrics = self._patch_body_shape_metrics(patch)
            if metrics is not None:
                circ = float(metrics["circularity"])
                tip_ratio = float(metrics["tip_ratio"])
                if circ >= 0.55 and tip_ratio < 1.45:
                    return None
        return self._score_template_anchor(
            match_val=match_val,
            ax=ax,
            ay=ay,
            width=width,
            height=height,
        )

    def _find_mining_node_blobs(
        self,
        gray: np.ndarray,
        search_mask: np.ndarray,
    ) -> list[tuple[float, float, float, float]]:
        """Circulos brancos do minimapa (nos de mineracao)."""
        blobs: list[tuple[float, float, float, float]] = []
        bright = cv2.inRange(gray, 112, 255)
        bright = cv2.bitwise_and(bright, search_mask)
        kernel = np.ones((2, 2), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 10.0 or area > 420.0:
                continue
            peri = float(cv2.arcLength(contour, True))
            if peri < 1.0:
                continue
            circularity = float(4.0 * math.pi * area / (peri * peri))
            if circularity < 0.65:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] < 1.0:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            radius = math.sqrt(area / math.pi)
            blobs.append((cx, cy, radius, circularity))

        return blobs

    def _point_on_mining_node(
        self,
        x: float,
        y: float,
        blobs: list[tuple[float, float, float, float]],
        *,
        margin_px: float = 10.0,
        min_node_radius: float = 0.0,
    ) -> bool:
        for nx, ny, radius, _circ in blobs:
            if radius < min_node_radius:
                continue
            if math.hypot(x - nx, y - ny) <= radius + margin_px:
                return True
        return False

    def _reject_circle_node_at(self, gray: np.ndarray, cx: int, cy: int) -> bool:
        """Nó branco = blob redondo; seta = baixa circularidade ou perde p/ template circular."""
        half = 8
        if not (half <= cx < gray.shape[1] - half and half <= cy < gray.shape[0] - half):
            return False
        patch = gray[cy - half : cy + half + 1, cx - half : cx + half + 1]
        circle_val = self._circle_template_match_at(gray, cx, cy)
        metrics = self._patch_body_shape_metrics(patch)
        if metrics is None:
            return circle_val >= 0.38
        circ = float(metrics["circularity"])
        tip_ratio = float(metrics["tip_ratio"])
        vertices = int(metrics["vertices"])
        if tip_ratio >= 2.5:
            return False
        if circ >= 0.72 and circle_val >= 0.38:
            return True
        if tip_ratio >= 1.45 and circ <= 0.72:
            return False
        if circ >= 0.72 and circle_val < 0.32 and tip_ratio >= 1.45:
            return False
        if circ >= 0.64 and circle_val >= 0.28:
            return True
        if circle_val >= 0.40 and circ >= 0.48:
            return True
        if circ >= 0.70 and tip_ratio < 2.5 and vertices >= 6:
            return True
        return self._is_obvious_circle_patch(patch)

    def _arrow_beats_circle(self, gray: np.ndarray, cx: int, cy: int) -> bool:
        return self._arrow_beats_circle_vals(
            self._template_match_at(gray, cx, cy),
            gray,
            cx,
            cy,
        )

    def _arrow_beats_circle_vals(
        self,
        arrow_val: float,
        gray: np.ndarray,
        cx: int,
        cy: int,
    ) -> bool:
        circle_val = self._circle_template_match_at(gray, cx, cy)
        half = 8
        if half <= cx < gray.shape[1] - half and half <= cy < gray.shape[0] - half:
            patch = gray[cy - half : cy + half + 1, cx - half : cx + half + 1]
            metrics = self._patch_body_shape_metrics(patch)
            if metrics is not None and float(metrics["circularity"]) >= 0.58:
                if circle_val >= 0.26:
                    return arrow_val >= max(0.58, circle_val + 0.22)
        if circle_val >= 0.35:
            return arrow_val >= 0.46 and arrow_val >= circle_val + 0.14
        if circle_val < 0.26:
            return arrow_val >= 0.20 or arrow_val >= circle_val + 0.02
        return arrow_val >= 0.38 and arrow_val >= circle_val + 0.08

    def _is_obvious_circle_patch(self, patch: np.ndarray) -> bool:
        metrics = self._patch_body_shape_metrics(patch)
        if metrics is None:
            return False
        circ = float(metrics["circularity"])
        tip_ratio = float(metrics["tip_ratio"])
        vertices = int(metrics["vertices"])
        return (circ >= 0.72 and tip_ratio < 1.25) or (vertices >= 9 and circ >= 0.58)

    def _template_match_at(self, gray: np.ndarray, cx: int, cy: int) -> float:
        best = 0.0
        for _angle, tmpl, _pvx, _pvy, _tvx, _tvy in self._rotated_arrow_templates():
            th, tw = tmpl.shape
            x0 = int(round(cx - tw * 0.5))
            y0 = int(round(cy - th * 0.5))
            if x0 < 0 or y0 < 0 or x0 + tw > gray.shape[1] or y0 + th > gray.shape[0]:
                continue
            patch = gray[y0 : y0 + th, x0 : x0 + tw]
            val = float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            best = max(best, val)
        return best

    def _patch_body_mask(self, patch: np.ndarray) -> np.ndarray:
        return (patch >= 95) & (patch <= 215)

    def _patch_body_shape_metrics(self, patch: np.ndarray) -> dict[str, float | bool] | None:
        body = self._patch_body_mask(patch)
        ys, xs = np.where(body)
        if xs.size < 6:
            return None

        body_u8 = body.astype(np.uint8) * 255
        contours, _ = cv2.findContours(body_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 10.0:
            return None

        peri = float(cv2.arcLength(contour, True))
        circularity = 0.0
        if peri > 1.0:
            circularity = float(4.0 * math.pi * area / (peri * peri))

        cx_f = float(xs.mean())
        cy_f = float(ys.mean())
        dists = np.hypot(xs.astype(np.float64) - cx_f, ys.astype(np.float64) - cy_f)
        tip_ratio = float(dists.max() / max(float(dists.mean()), 1e-3))

        pixels = patch[ys, xs]
        std = float(pixels.std()) if pixels.size else 0.0

        h, w = patch.shape
        dark = patch < 88
        dark_asym = max(
            abs(float(dark[:, : w // 2].mean()) - float(dark[:, w // 2 :].mean())),
            abs(float(dark[: h // 2, :].mean()) - float(dark[h // 2 :, :].mean())),
        )

        white = patch >= 120
        gray_part = (patch >= 90) & (patch < 120)
        has_bicolor = bool(white.any() and gray_part.any())

        epsilon = max(0.035 * peri, 1.0)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        return {
            "circularity": circularity,
            "tip_ratio": tip_ratio,
            "std": std,
            "dark_asym": dark_asym,
            "has_bicolor": has_bicolor,
            "vertices": float(vertices),
        }

    def _patch_rejects_mining_node(
        self,
        patch: np.ndarray,
        *,
        gray: np.ndarray | None = None,
        cx: int | None = None,
        cy: int | None = None,
    ) -> bool:
        """Nó de mineração = blob redondo e uniforme; seta = pontiaguda e assimétrica."""
        metrics = self._patch_body_shape_metrics(patch)
        if metrics is None:
            return True

        circ = float(metrics["circularity"])
        tip_ratio = float(metrics["tip_ratio"])
        std = float(metrics["std"])
        dark_asym = float(metrics["dark_asym"])
        has_bicolor = bool(metrics["has_bicolor"])
        vertices = int(metrics["vertices"])

        if gray is not None and cx is not None and cy is not None:
            arrow_val = self._template_match_at(gray, cx, cy)
            circle_val = self._circle_template_match_at(gray, cx, cy)
            if circle_val >= arrow_val + 0.06:
                return True
            if arrow_val >= 0.48 and arrow_val >= circle_val + 0.12:
                return False

        if vertices >= 8 and circ >= 0.58:
            return True
        if circ >= 0.68:
            return True

        arrow_like = (
            tip_ratio >= 1.35
            and circ <= 0.62
            and vertices <= 7
            and (dark_asym >= 0.20 or has_bicolor)
        )
        if arrow_like:
            return False

        if circ >= 0.58 and tip_ratio < 1.30:
            return True
        if circ >= 0.52 and std < 10.0 and tip_ratio < 1.35 and dark_asym < 0.20:
            return True
        if not has_bicolor and circ >= 0.48 and tip_ratio < 1.32:
            return True
        return tip_ratio < 1.22

    def _template_candidates(
        self,
        facing_deg: float | None = None,
        *,
        span_deg: float = 360.0,
    ) -> list[tuple[float, np.ndarray, float, float, float, float]]:
        templates = self._template_sets()
        if facing_deg is None or span_deg >= 359.0:
            return templates
        base = (float(facing_deg) + 90.0) % 360.0
        half = span_deg * 0.5
        picked: list[tuple[float, np.ndarray, float, float, float, float]] = []
        for angle, tmpl, pvx, pvy, tvx, tvy in templates:
            delta = (angle - base + 180.0) % 360.0 - 180.0
            if abs(delta) <= half:
                picked.append((angle, tmpl, pvx, pvy, tvx, tvy))
        return picked or templates

    def _pack_template_hit(
        self,
        *,
        max_loc: tuple[int, int],
        tmpl: np.ndarray,
        pivot_x: float,
        pivot_y: float,
        tip_x_t: float,
        tip_y_t: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        ax = offset_x + max_loc[0] + pivot_x
        ay = offset_y + max_loc[1] + pivot_y
        tip_x = offset_x + max_loc[0] + tip_x_t
        tip_y = offset_y + max_loc[1] + tip_y_t
        angle = math.degrees(math.atan2(tip_y - ay, tip_x - ax))
        return tip_x, tip_y, angle, ax, ay

    def _find_arrow_by_template(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
        *,
        roi: tuple[int, int, int, int] | None = None,
        facing_deg: float | None = None,
        min_match: float = 0.40,
        relax: bool = False,
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        height, width = gray.shape[:2]
        if roi is None:
            x0, y0, x1, y1 = 0, 0, width, height
            view = gray
        else:
            x0, y0, x1, y1 = roi
            view = gray[y0:y1, x0:x1]

        best_val = min_match
        best: tuple[float, float, float, float, float] | None = None
        best_hit: tuple[float, np.ndarray, float, float, float, float, float, tuple[int, int], float] | None = None

        for angle, tmpl, pvx, pvy, tvx, tvy in self._template_candidates(
            facing_deg, span_deg=90.0 if facing_deg else 360.0
        ):
            th, tw = tmpl.shape
            if th >= view.shape[0] or tw >= view.shape[1]:
                continue
            res = cv2.matchTemplate(view, tmpl, cv2.TM_CCOEFF_NORMED)
            for max_val, max_loc in self._iter_template_peaks(res, min_match=min_match):
                th, tw = tmpl.shape
                probe_x = x0 + max_loc[0] + tw * 0.5
                probe_y = y0 + max_loc[1] + th * 0.5
                ax = x0 + max_loc[0] + pvx
                ay = y0 + max_loc[1] + pvy
                score = self._accept_template_anchor(
                    gray,
                    hsv,
                    search,
                    node_blobs,
                    ax=ax,
                    ay=ay,
                    probe_x=probe_x,
                    probe_y=probe_y,
                    match_val=max_val,
                    width=width,
                    height=height,
                    relax=relax,
                )
                if score is None or score <= best_val:
                    continue
                best_val = score
                best_hit = (angle, tmpl, float(x0), float(y0), pvx, pvy, tvx, tvy, max_loc, max_val)

        if best_hit is None:
            return None
        angle, tmpl, offset_x, offset_y, pvx, pvy, tvx, tvy, max_loc, max_val = best_hit
        best = self._pack_template_hit(
            max_loc=max_loc,
            tmpl=tmpl,
            pivot_x=pvx,
            pivot_y=pvy,
            tip_x_t=tvx,
            tip_y_t=tvy,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        tip_x, tip_y, angle, anchor_x, anchor_y = best
        contour = np.array([[[int(anchor_x), int(anchor_y)]]], dtype=np.int32)
        return tip_x, tip_y, angle, anchor_x, anchor_y, contour

    def _find_arrow_around(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
        cx: float,
        cy: float,
        radius: float,
        *,
        relax: bool = False,
        min_match: float = 0.36,
        facing_deg: float | None = None,
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        width, height = gray.shape[1], gray.shape[0]
        r = int(max(radius, 24))
        x0 = max(0, int(cx) - r)
        y0 = max(0, int(cy) - r)
        x1 = min(width, int(cx) + r)
        y1 = min(height, int(cy) + r)
        return self._find_arrow_by_template(
            gray,
            hsv,
            search,
            node_blobs,
            roi=(x0, y0, x1, y1),
            facing_deg=facing_deg if facing_deg is not None else self._last_facing_deg,
            min_match=min_match,
            relax=relax,
        )

    def _find_arrow_near_hint(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        height, width = gray.shape[:2]
        hx = width * self.player_center_ratio[0]
        hy = height * self.player_center_ratio[1]
        radius = max(self.arrow_search_radius_px, 36)
        return self._find_arrow_around(
            gray,
            hsv,
            search,
            node_blobs,
            hx,
            hy,
            radius,
            min_match=0.36,
        )

    def _find_arrow_near_last(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        if self._last_anchor_x is None or self._last_anchor_y is None:
            return None
        radius = max(self.arrow_search_radius_px, 72)
        return self._find_arrow_around(
            gray,
            hsv,
            search,
            node_blobs,
            self._last_anchor_x,
            self._last_anchor_y,
            radius,
            relax=True,
            min_match=0.30,
        )

    def _find_arrow_grid_near(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
        cx: float,
        cy: float,
        radius: float,
        *,
        min_match: float = 0.28,
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        """Fallback: template centrado em cada ponto (funciona quando slide falha)."""
        height, width = gray.shape[:2]
        r = int(max(radius, 24))
        x_min = max(8, int(cx) - r)
        x_max = min(width - 8, int(cx) + r)
        y_min = max(8, int(cy) - r)
        y_max = min(height - 8, int(cy) + r)

        best_val = min_match
        best_pack: tuple[float, float, float, float, float] | None = None

        for iy in range(y_min, y_max, 3):
            for ix in range(x_min, x_max, 3):
                if search[iy, ix] == 0:
                    continue
                match_val = 0.0
                best_tpl: tuple[float, np.ndarray, float, float, float, float] | None = None
                for angle, tmpl, pvx, pvy, tvx, tvy in self._template_candidates(
                    self._last_facing_deg, span_deg=360.0
                ):
                    th, tw = tmpl.shape
                    x0 = int(round(ix - tw * 0.5))
                    y0 = int(round(iy - th * 0.5))
                    if x0 < 0 or y0 < 0 or x0 + tw > width or y0 + th > height:
                        continue
                    patch = gray[y0 : y0 + th, x0 : x0 + tw]
                    val = float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                    if val <= match_val:
                        continue
                    match_val = val
                    best_tpl = (angle, tmpl, pvx, pvy, tvx, tvy)

                if best_tpl is None or match_val < min_match:
                    continue

                _angle, tmpl, pvx, pvy, tvx, tvy = best_tpl
                th, tw = tmpl.shape
                x0 = int(round(ix - tw * 0.5))
                y0 = int(round(iy - th * 0.5))
                ax = float(x0) + pvx
                ay = float(y0) + pvy
                score = self._accept_template_anchor(
                    gray,
                    hsv,
                    search,
                    node_blobs,
                    ax=ax,
                    ay=ay,
                    probe_x=float(ix),
                    probe_y=float(iy),
                    match_val=match_val,
                    width=width,
                    height=height,
                    relax=True,
                )
                if score is None:
                    continue
                tip_x = float(x0) + tvx
                tip_y = float(y0) + tvy
                if not self._validate_arrow_anchor(
                    gray, hsv, ax, ay, node_blobs, tip_x=tip_x, tip_y=tip_y
                ):
                    continue
                if score <= best_val:
                    continue
                best_val = score
                angle = math.degrees(math.atan2(tip_y - ay, tip_x - ax))
                best_pack = (tip_x, tip_y, angle, ax, ay)

        if best_pack is None:
            return None
        tip_x, tip_y, angle, anchor_x, anchor_y = best_pack
        contour = np.array([[[int(anchor_x), int(anchor_y)]]], dtype=np.int32)
        return tip_x, tip_y, angle, anchor_x, anchor_y, contour

    def _find_arrow_local(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        if self._last_anchor_x is None or self._last_anchor_y is None:
            return None
        radius = max(self.arrow_search_radius_px * 0.55, 48)
        return self._find_arrow_around(
            gray,
            hsv,
            search,
            node_blobs,
            self._last_anchor_x,
            self._last_anchor_y,
            radius,
            relax=True,
            min_match=0.30,
        )

    def reset(self) -> None:
        self._smooth_tip_x = None
        self._smooth_tip_y = None
        self._last_facing_deg = None
        self._player_lost_frames = 0
        self._last_arrow_contour = None
        self._last_anchor_x = None
        self._last_anchor_y = None
        self._node_blob_cache = None
        self._node_blob_frame = -999
        self._detect_frame = 0
        self._auto_anchor_x = None
        self._auto_anchor_y = None

    @property
    def last_facing_deg(self) -> float | None:
        return self._last_facing_deg

    def player_position(self, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        """
        Meio da seta no GTA/Grand RP: fixo no minimapa.
        Ideia: base do disco do minimapa → sobe no eixo vertical central → seta.
        (O mapa gira; a seta nao anda na tela.)
        """
        height, width = frame_shape[:2]
        if self.fixed_player_anchor:
            if self.player_center_calibrated:
                return (
                    width * self.player_center_ratio[0],
                    height * self.player_center_ratio[1],
                )
            if self._auto_anchor_x is not None and self._auto_anchor_y is not None:
                return self._auto_anchor_x, self._auto_anchor_y
            # Fundo do disco circular → sobe no eixo central.
            disc_cx = width * 0.5
            disc_cy = height * 0.5
            disc_r = min(width, height) * 0.48
            disc_bottom = disc_cy + disc_r
            # Seta fica ~55% do raio acima do fundo do disco (acima da barra de vida).
            arrow_y = disc_bottom - disc_r * 0.55
            arrow_y = min(max(arrow_y, height * 0.55), height * 0.82)
            return disc_cx, arrow_y
        if self._last_anchor_x is not None and self._last_anchor_y is not None:
            return self._last_anchor_x, self._last_anchor_y
        return (
            width * self.player_center_ratio[0],
            height * self.player_center_ratio[1],
        )

    def _estimate_player_anchor(self, frame_bgr: np.ndarray) -> tuple[float, float]:
        """Varre o eixo vertical do minimapa e acha onde a seta encaixa melhor."""
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cx = width * 0.5
        # Grand RP / GTA: seta fica na metade inferior (barra de vida embaixo).
        y_min = int(height * 0.52)
        y_max = int(height * 0.86)
        best_score = 0.30
        best_x, best_y = cx, height * 0.72

        for iy in range(y_min, y_max, 3):
            ax, ay = cx, float(iy)
            ix, iy_i = int(round(ax)), int(round(ay))
            if not (0 <= ix < width and 0 <= iy_i < height):
                continue
            if self._reject_circle_node_at(gray, ix, iy_i):
                continue
            peak = 0.0
            for _angle, tmpl, pvx, pvy, _tvx, _tvy in self._template_sets():
                th, tw = tmpl.shape
                x0 = int(round(ax - pvx))
                y0 = int(round(ay - pvy))
                if x0 < 0 or y0 < 0 or x0 + tw > width or y0 + th > height:
                    continue
                roi = gray[y0 : y0 + th, x0 : x0 + tw]
                if roi.shape[:2] != tmpl.shape[:2]:
                    continue
                val = float(cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                peak = max(peak, val)
            y_bonus = max(0.0, (iy / max(height, 1) - 0.55)) * 0.12
            score = peak + y_bonus
            if score > best_score:
                best_score = score
                best_x, best_y = ax, ay

        return best_x, best_y

    @staticmethod
    def disc_center(frame_shape: tuple[int, ...]) -> tuple[float, float]:
        """Legado — centro geometrico do ROI (nao e a seta no Grand RP)."""
        height, width = frame_shape[:2]
        return width * 0.5, height * 0.5

    def _minimap_disc_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        cx = int(width * 0.5)
        cy = int(height * 0.5)
        radius = int(min(width, height) * 0.48)
        cv2.circle(mask, (cx, cy), max(radius, 1), 255, -1)
        return mask

    def _ui_exclusion_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.full((height, width), 255, dtype=np.uint8)
        cv2.rectangle(mask, (0, int(height * 0.72)), (int(width * 0.45), height), 0, -1)
        cv2.rectangle(mask, (int(width * 0.62), 0), (width, int(height * 0.35)), 0, -1)
        margin = max(int(min(width, height) * 0.04), 2)
        cv2.rectangle(mask, (0, 0), (width, margin), 0, -1)
        cv2.rectangle(mask, (0, 0), (margin, height), 0, -1)
        cv2.rectangle(mask, (width - margin, 0), (width, height), 0, -1)
        return mask

    def _search_mask(self, width: int, height: int, anchor_x: float, anchor_y: float) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            mask,
            (int(anchor_x), int(anchor_y)),
            max(int(self.arrow_search_radius_px), 1),
            255,
            -1,
        )
        return mask

    def _arrow_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Mascara da seta bicolor: branco+cinza compactos sobre fundo escuro."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        low_sat = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 95, 255]))

        white_lo = max(self.arrow_white_min - 40, 118)
        white = cv2.inRange(gray, white_lo, 255)
        mid_gray = cv2.inRange(gray, self.arrow_gray_min, self.arrow_gray_max)
        body = cv2.bitwise_or(white, mid_gray)
        body = cv2.bitwise_and(body, low_sat)

        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        # Fechamento curto: junta as duas metades sem colar nas estradas.
        merged = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k3)
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k3)

        # Seta fica sobre fundo escuro (~55-70); estrada uniforme nao tem vizinho tao escuro.
        dark_bg = cv2.inRange(gray, 0, 82)
        dark_near = cv2.dilate(dark_bg, k5, iterations=1)
        on_dark = cv2.bitwise_and(merged, dark_near)

        # Regiao onde branco e cinza ficam colados (assinatura bicolor).
        adjacency = cv2.bitwise_and(
            cv2.dilate(white, k3, iterations=1),
            cv2.dilate(mid_gray, k3, iterations=1),
        )

        mask = cv2.bitwise_or(on_dark, adjacency)
        mask = cv2.bitwise_and(mask, low_sat)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    def _two_tone_score(self, gray: np.ndarray, contour: np.ndarray) -> float:
        """Bonus quando o contorno tem pixels brancos E cinza (seta bicolor)."""
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            return 0.0
        patch = gray[y : y + h, x : x + w]
        local = np.zeros(patch.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            local,
            [contour - np.array([[x, y]])],
            -1,
            255,
            -1,
        )
        pixels = patch[local > 0]
        if pixels.size < 4:
            return 0.0
        has_white = bool(np.any(pixels >= max(self.arrow_white_min - 40, 118)))
        has_gray = bool(np.any((pixels >= self.arrow_gray_min) & (pixels < max(self.arrow_white_min - 40, 118))))
        if has_white and has_gray:
            return 120.0
        if has_white or has_gray:
            return 25.0
        return 0.0

    def _patch_arrow_score(self, patch: np.ndarray) -> float:
        """Pontua um recorte local: seta = bicolor compacta sobre fundo escuro."""
        if patch.shape[0] < 9 or patch.shape[1] < 9:
            return 0.0
        if self._patch_rejects_mining_node(patch):
            return 0.0

        metrics = self._patch_body_shape_metrics(patch)
        if metrics is None:
            return 0.0

        dark = patch < 88
        body = self._patch_body_mask(patch)
        white = patch >= max(self.arrow_white_min - 35, 120)

        dark_frac = float(np.count_nonzero(dark)) / patch.size
        body_frac = float(np.count_nonzero(body)) / patch.size
        tip_ratio = float(metrics["tip_ratio"])
        dark_asym = float(metrics["dark_asym"])
        strong_arrow = tip_ratio >= 1.65 and dark_asym >= 0.45 and bool(metrics["has_bicolor"])

        min_body = 0.30 if strong_arrow else 0.55
        max_grad = 68.0 if strong_arrow else 42.0
        if dark_frac < 0.08 or dark_frac > 0.72:
            return 0.0
        if body_frac < min_body or body_frac > 0.88:
            return 0.0

        has_white = bool(np.any(white))
        has_gray = bool(np.any(body & ~white))
        if not (has_white or has_gray):
            return 0.0
        bicolor_bonus = 55.0 if has_white and has_gray else 12.0

        h, w = patch.shape
        left_mean = float(patch[:, : w // 2].mean())
        right_mean = float(patch[:, w // 2 :].mean())
        top_mean = float(patch[: h // 2, :].mean())
        bot_mean = float(patch[h // 2 :, :].mean())
        grad = max(abs(right_mean - left_mean), abs(bot_mean - top_mean))
        if grad < 12.0 or grad > max_grad:
            return 0.0

        ys, xs = np.where(body)
        span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        if span > 18.0:
            return 0.0

        tip_ratio = float(metrics["tip_ratio"])
        dark_asym = float(metrics["dark_asym"])
        circ = float(metrics["circularity"])
        shape_bonus = (tip_ratio - 1.0) * 28.0 + dark_asym * 45.0 - circ * 35.0
        if bool(metrics["has_bicolor"]):
            shape_bonus += 18.0

        return bicolor_bonus + grad * 1.4 + body_frac * 40.0 - span * 2.5 + shape_bonus

    def _tip_from_patch(
        self,
        patch: np.ndarray,
        origin_x: int,
        origin_y: int,
    ) -> tuple[float, float, float, float, float] | None:
        body = self._patch_body_mask(patch).astype(np.uint8) * 255
        if int(np.count_nonzero(body)) < 5:
            return None

        moments = cv2.moments(body, binaryImage=True)
        if moments["m00"] < 1.0:
            return None
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

        ys, xs = np.where(body > 0)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        vecs = pts - np.array([cx, cy], dtype=np.float64)
        dists = np.linalg.norm(vecs, axis=1)
        idx = int(dists.argmax())
        tip_dist = float(dists[idx])
        if tip_dist < self.arrow_min_tip_dist_px or tip_dist > self.arrow_max_tip_dist_px:
            return None

        anchor_x = origin_x + cx
        anchor_y = origin_y + cy
        tip_x = origin_x + float(pts[idx, 0])
        tip_y = origin_y + float(pts[idx, 1])
        angle = math.degrees(math.atan2(tip_y - anchor_y, tip_x - anchor_x))
        return tip_x, tip_y, angle, anchor_x, anchor_y

    def _find_arrow_by_patch(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        """Varredura local por padrao bicolor (nao depende de contorno unico)."""
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        search = self._ui_exclusion_mask(width, height)
        search = cv2.bitwise_and(search, self._minimap_disc_mask(width, height))
        node_blobs = self._find_mining_node_blobs(gray, search)

        half = 10
        step = 2
        best_score = 35.0
        best: tuple[float, float, float, float, float] | None = None

        for cy in range(half, height - half, step):
            for cx in range(half, width - half, step):
                if search[cy, cx] == 0:
                    continue
                if self._point_on_mining_node(float(cx), float(cy), node_blobs):
                    continue
                if not self._arrow_beats_circle(gray, cx, cy):
                    continue
                patch = gray[cy - half : cy + half + 1, cx - half : cx + half + 1]
                if self._patch_rejects_mining_node(patch, gray=gray, cx=cx, cy=cy):
                    continue
                score = self._patch_arrow_score(patch)
                if score <= best_score:
                    continue
                tmpl_val = self._template_match_at(gray, cx, cy)
                score += tmpl_val * 45.0
                score += self._template_prior_bonus(cx, cy, width, height) * 50.0
                tip_data = self._tip_from_patch(patch, cx - half, cy - half)
                if tip_data is None:
                    continue
                if self._smooth_tip_x is not None:
                    tip_x, tip_y, _, ax, ay = tip_data
                    score += max(
                        0.0,
                        24.0
                        - math.hypot(tip_x - self._smooth_tip_x, tip_y - self._smooth_tip_y),
                    )
                best_score = score
                best = tip_data

        if best is None:
            return None
        tip_x, tip_y, angle, anchor_x, anchor_y = best
        contour = np.array(
            [[[int(anchor_x), int(anchor_y)]]],
            dtype=np.int32,
        )
        return tip_x, tip_y, angle, anchor_x, anchor_y, contour

    def _tip_from_contour_global(
        self,
        contour: np.ndarray,
    ) -> tuple[float, float, float, float, float] | None:
        """Ponta = vertice mais distante do centroide (funciona com seta bicolor)."""
        area = float(cv2.contourArea(contour))
        if area < self.arrow_min_area or area > self.arrow_max_area:
            return None

        centroid = self._contour_centroid(contour)
        if centroid is None:
            return None
        anchor_x, anchor_y = centroid

        pts = contour.reshape(-1, 2).astype(np.float64)
        vecs = pts - np.array([anchor_x, anchor_y], dtype=np.float64)
        dists = np.linalg.norm(vecs, axis=1)
        if dists.size < 3:
            return None

        idx = int(dists.argmax())
        tip_dist = float(dists[idx])
        if tip_dist < self.arrow_min_tip_dist_px or tip_dist > self.arrow_max_tip_dist_px:
            return None

        tip_x = float(pts[idx, 0])
        tip_y = float(pts[idx, 1])
        angle = math.degrees(math.atan2(vecs[idx, 1], vecs[idx, 0]))
        return tip_x, tip_y, angle, anchor_x, anchor_y

    def _tip_from_contour(
        self,
        contour: np.ndarray,
        anchor_x: float,
        anchor_y: float,
    ) -> tuple[float, float, float] | None:
        global_tip = self._tip_from_contour_global(contour)
        if global_tip is None:
            return None
        tip_x, tip_y, angle, cx, cy = global_tip
        if math.hypot(cx - anchor_x, cy - anchor_y) > self.arrow_max_centroid_dist_px:
            return None
        return tip_x, tip_y, angle

    def _validate_arrow_anchor(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        anchor_x: float,
        anchor_y: float,
        node_blobs: list[tuple[float, float, float, float]],
        *,
        tip_x: float | None = None,
        tip_y: float | None = None,
    ) -> bool:
        if self._point_on_mining_node(anchor_x, anchor_y, node_blobs, min_node_radius=4.0):
            return False
        if tip_x is not None and tip_y is not None:
            probe_x = anchor_x + (tip_x - anchor_x) * 0.38
            probe_y = anchor_y + (tip_y - anchor_y) * 0.38
        else:
            probe_x, probe_y = anchor_x, anchor_y
        ax, ay = int(round(probe_x)), int(round(probe_y))
        if self._reject_circle_node_at(gray, ax, ay):
            return False
        tracking = self._near_last_anchor(anchor_x, anchor_y)
        if not tracking and self._patch_has_mining_color(hsv, ax, ay):
            return False
        if not self._is_achromatic_at(hsv, ax, ay):
            return False
        if not tracking and not self._has_bicolor_arrow_patch(gray, ax, ay):
            return False
        half = 8
        if not (half <= ax < gray.shape[1] - half and half <= ay < gray.shape[0] - half):
            return True
        patch = gray[ay - half : ay + half + 1, ax - half : ax + half + 1]
        metrics = self._patch_body_shape_metrics(patch)
        if metrics is None:
            return True
        circ = float(metrics["circularity"])
        tip_ratio = float(metrics["tip_ratio"])
        return not (circ >= 0.55 and tip_ratio < 1.45)

    def _analyze_arrow_contour(
        self,
        contour: np.ndarray,
        gray: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]] | None = None,
        hsv: np.ndarray | None = None,
    ) -> tuple[float, float, float, float, float, float] | None:
        """Retorna (score, tip_x, tip_y, angle, anchor_x, anchor_y). Menor score = melhor."""
        global_tip = self._tip_from_contour_global(contour)
        if global_tip is None:
            return None
        tip_x, tip_y, angle, anchor_x, anchor_y = global_tip

        if node_blobs is not None and hsv is not None and not self._validate_arrow_anchor(
            gray, hsv, anchor_x, anchor_y, node_blobs
        ):
            return None

        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0
        if perimeter > 1.0:
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity > 0.78:
            return None

        two_tone = self._two_tone_score(gray, contour)
        if two_tone < 80.0:
            return None

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > 28 or bh > 28:
            return None

        tip_dist = math.hypot(tip_x - anchor_x, tip_y - anchor_y)
        score = -two_tone + circularity * 80.0 + max(area - 140.0, 0.0) * 0.8 + tip_dist * 0.15
        score += max(bw, bh) * 1.5
        if self._smooth_tip_x is not None:
            score += math.hypot(tip_x - self._smooth_tip_x, tip_y - self._smooth_tip_y) * 0.25
        return score, tip_x, tip_y, angle, anchor_x, anchor_y

    def _find_arrow_global(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        """Busca a seta bicolor em todo o disco do minimapa."""
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = self._arrow_mask(frame_bgr)
        mask = cv2.bitwise_and(mask, self._ui_exclusion_mask(width, height))
        mask = cv2.bitwise_and(mask, self._minimap_disc_mask(width, height))
        node_blobs = self._find_mining_node_blobs(gray, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = float("inf")
        for contour in contours:
            analyzed = self._analyze_arrow_contour(contour, gray, node_blobs)
            if analyzed is None:
                continue
            score, tip_x, tip_y, angle, anchor_x, anchor_y = analyzed
            if score >= best_score:
                continue
            best_score = score
            best = (tip_x, tip_y, angle, anchor_x, anchor_y, contour)
        return best

    def _find_arrow(
        self,
        frame_bgr: np.ndarray,
        anchor_x: float,
        anchor_y: float,
    ) -> tuple[float, float, float, np.ndarray] | None:
        height, width = frame_bgr.shape[:2]
        mask = self._arrow_mask(frame_bgr)
        mask = cv2.bitwise_and(mask, self._ui_exclusion_mask(width, height))
        mask = cv2.bitwise_and(mask, self._search_mask(width, height, anchor_x, anchor_y))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = float("inf")
        for contour in contours:
            tip_data = self._tip_from_contour(contour, anchor_x, anchor_y)
            if tip_data is None:
                continue
            tip_x, tip_y, _angle = tip_data
            moments = cv2.moments(contour)
            if moments["m00"] < 1.0:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            centroid_dist = math.hypot(cx - anchor_x, cy - anchor_y)
            tip_dist = math.hypot(tip_x - anchor_x, tip_y - anchor_y)
            score = centroid_dist * 2.0 + tip_dist
            if self._smooth_tip_x is not None:
                score += math.hypot(tip_x - self._smooth_tip_x, tip_y - self._smooth_tip_y) * 0.5
            if score >= best_score:
                continue
            best_score = score
            best = (tip_x, tip_y, _angle, contour)

        return best

    def _radial_fallback(
        self,
        frame_bgr: np.ndarray,
        anchor_x: float,
        anchor_y: float,
        *,
        node_blobs: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[float, float, float] | None:
        """Fallback: pixels claros ao redor do ancora indicam a ponta da seta."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        min_r = max(self.arrow_min_tip_dist_px * 0.6, 4.0)
        max_r = self.arrow_max_tip_dist_px + 6.0
        half = int(max_r + 4)
        x0 = max(0, int(anchor_x) - half)
        x1 = min(width, int(anchor_x) + half + 1)
        y0 = max(0, int(anchor_y) - half)
        y1 = min(height, int(anchor_y) + half + 1)

        arrow_mask = self._arrow_mask(frame_bgr[y0:y1, x0:x1])
        ys, xs = np.where(arrow_mask > 0)
        if xs.size < 6:
            patch = gray[y0:y1, x0:x1]
            ys, xs = np.where(patch >= max(self.arrow_gray_min - 12, 95))
            if xs.size < 6:
                return None
            xs = xs + x0
            ys = ys + y0
        else:
            xs = xs + x0
            ys = ys + y0
        dx = xs.astype(np.float64) - anchor_x
        dy = ys.astype(np.float64) - anchor_y
        dist = np.hypot(dx, dy)
        keep = (dist >= min_r) & (dist <= max_r)
        if int(np.count_nonzero(keep)) < 6:
            return None
        dx, dy, dist = dx[keep], dy[keep], dist[keep]
        xs_k, ys_k = xs[keep], ys[keep]
        if node_blobs:
            node_keep = np.ones(xs_k.shape[0], dtype=bool)
            for idx, (px, py) in enumerate(zip(xs_k, ys_k)):
                if self._point_on_mining_node(float(px), float(py), node_blobs, min_node_radius=5.0):
                    node_keep[idx] = False
            if int(np.count_nonzero(node_keep)) < 4:
                return None
            dx, dy, dist = dx[node_keep], dy[node_keep], dist[node_keep]
            xs_k, ys_k = xs_k[node_keep], ys_k[node_keep]
        # Ponta = pixel da seta mais longe do pivot (nariz), nao media ponderada
        # (media puxa para a base larga e inverte ~180°).
        tip_idx = int(np.argmax(dist))
        tip_x = float(xs_k[tip_idx])
        tip_y = float(ys_k[tip_idx])
        tip_len = float(dist[tip_idx])
        if tip_len < self.arrow_min_tip_dist_px:
            return None
        tip_len = min(tip_len, self.arrow_max_tip_dist_px)
        ux = (tip_x - anchor_x) / max(float(dist[tip_idx]), 1e-3)
        uy = (tip_y - anchor_y) / max(float(dist[tip_idx]), 1e-3)
        tip_x = anchor_x + ux * tip_len
        tip_y = anchor_y + uy * tip_len
        angle = math.degrees(math.atan2(uy, ux))
        return tip_x, tip_y, angle

    def _accept_tip(
        self, tip_x: float, tip_y: float, anchor_x: float, anchor_y: float
    ) -> tuple[float, float] | None:
        dist = math.hypot(tip_x - anchor_x, tip_y - anchor_y)
        if dist < self.arrow_min_tip_dist_px or dist > self.arrow_max_tip_dist_px:
            if self._smooth_tip_x is not None:
                smooth_dist = math.hypot(
                    self._smooth_tip_x - anchor_x, self._smooth_tip_y - anchor_y
                )
                if (
                    self.arrow_min_tip_dist_px
                    <= smooth_dist
                    <= self.arrow_max_tip_dist_px
                ):
                    return self._smooth_tip_x, self._smooth_tip_y
            return None

        if self._smooth_tip_x is not None:
            jump = math.hypot(tip_x - self._smooth_tip_x, tip_y - self._smooth_tip_y)
            if jump > self.arrow_max_tip_jump_px:
                return self._smooth_tip_x, self._smooth_tip_y

        if self._smooth_tip_x is None:
            return tip_x, tip_y

        alpha = self.player_position_smoothing
        sx = alpha * tip_x + (1.0 - alpha) * self._smooth_tip_x
        sy = alpha * tip_y + (1.0 - alpha) * self._smooth_tip_y
        smooth_dist = math.hypot(sx - anchor_x, sy - anchor_y)
        if (
            smooth_dist < self.arrow_min_tip_dist_px
            or smooth_dist > self.arrow_max_tip_dist_px
        ):
            return tip_x, tip_y
        return sx, sy

    def _contour_centroid(self, contour: np.ndarray) -> tuple[float, float] | None:
        moments = cv2.moments(contour)
        if moments["m00"] < 1.0:
            return None
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    def _angle_diff_deg(self, a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _resolve_tip_facing(
        self,
        frame_bgr: np.ndarray,
        anchor_x: float,
        anchor_y: float,
        tip_x: float,
        tip_y: float,
    ) -> tuple[float, float, float | None]:
        """Evita ponta invertida (180 graus) usando pixels claros ao redor."""
        dx = tip_x - anchor_x
        dy = tip_y - anchor_y
        dist = math.hypot(dx, dy)
        if dist < self.arrow_min_tip_dist_px:
            return tip_x, tip_y, None

        facing_a = math.degrees(math.atan2(dy, dx))
        flip_x = anchor_x - dx
        flip_y = anchor_y - dy
        facing_b = math.degrees(math.atan2(flip_y - anchor_y, flip_x - anchor_x))

        radial = self._radial_fallback(frame_bgr, anchor_x, anchor_y)
        if radial is not None:
            _, _, radial_facing = radial
            if self._angle_diff_deg(facing_b, radial_facing) < self._angle_diff_deg(
                facing_a, radial_facing
            ):
                return flip_x, flip_y, radial_facing
            return tip_x, tip_y, radial_facing

        return tip_x, tip_y, facing_a

    def _facing_from_tip(
        self, tip_x: float, tip_y: float, anchor_x: float, anchor_y: float
    ) -> float | None:
        dist = math.hypot(tip_x - anchor_x, tip_y - anchor_y)
        if dist < self.arrow_min_tip_dist_px or dist > self.arrow_max_tip_dist_px:
            return None
        return math.degrees(math.atan2(tip_y - anchor_y, tip_x - anchor_x))

    def _find_facing_at_fixed_anchor(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        search: np.ndarray,
        node_blobs: list[tuple[float, float, float, float]],
        anchor_x: float,
        anchor_y: float,
    ) -> tuple[float, float, float, float, float, np.ndarray] | None:
        """Template match centrado no pivot fixo — so detecta rotacao."""
        height, width = gray.shape[:2]
        ix, iy = int(round(anchor_x)), int(round(anchor_y))
        if not (0 <= ix < width and 0 <= iy < height):
            return None

        best_val = 0.32
        best_hit: tuple[float, float, float, float, float] | None = None
        facing_hint = self._last_facing_deg

        for angle, tmpl, pvx, pvy, tvx, tvy in self._template_candidates(
            facing_hint, span_deg=360.0
        ):
            th, tw = tmpl.shape
            x0 = int(round(anchor_x - pvx))
            y0 = int(round(anchor_y - pvy))
            if x0 < 0 or y0 < 0 or x0 + tw > width or y0 + th > height:
                continue
            pxi, pyi = int(round(anchor_x)), int(round(anchor_y))
            if not (0 <= pxi < width and 0 <= pyi < height):
                continue
            roi = gray[y0 : y0 + th, x0 : x0 + tw]
            if roi.shape[:2] != tmpl.shape[:2]:
                continue
            match_val = float(cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            if match_val < 0.32:
                continue
            if self._point_on_mining_node(anchor_x, anchor_y, node_blobs, min_node_radius=4.0):
                continue
            if self._reject_circle_node_at(gray, pxi, pyi):
                continue

            tip_x = x0 + tvx
            tip_y = y0 + tvy
            candidate_facing = math.degrees(
                math.atan2(tip_y - anchor_y, tip_x - anchor_x)
            )
            score = match_val
            if facing_hint is not None:
                delta = abs(
                    (candidate_facing - facing_hint + 180.0) % 360.0 - 180.0
                )
                score += max(0.0, 0.10 - delta / 40.0) * 0.35
            if score <= best_val:
                continue
            best_val = score
            best_hit = (tip_x, tip_y, candidate_facing, anchor_x, anchor_y)

        if best_hit is None:
            return None
        tip_x, tip_y, angle, ax, ay = best_hit
        contour = np.array([[[int(ax), int(ay)]]], dtype=np.int32)
        return tip_x, tip_y, angle, ax, ay, contour

    def _bicolor_tip_from_mask(
        self,
        frame_bgr: np.ndarray,
        anchor_x: float,
        anchor_y: float,
    ) -> tuple[float, float, float] | None:
        """
        Nariz via padrao bicolor da seta GTA:
          - metade branca | metade cinza
          - contorno preto
          - a divisoria branco/cinza aponta para o nariz
        Frente = rotacao 90° CCW do vetor cinza→branco (branco fica a esquerda).
        """
        height, width = frame_bgr.shape[:2]
        half = int(self.arrow_max_tip_dist_px + 12)
        x0 = max(0, int(anchor_x) - half)
        x1 = min(width, int(anchor_x) + half + 1)
        y0 = max(0, int(anchor_y) - half)
        y1 = min(height, int(anchor_y) + half + 1)
        patch = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        low_sat = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 100, 255]))

        white_lo = max(self.arrow_white_min - 10, 155)
        white = cv2.bitwise_and(cv2.inRange(gray, white_lo, 255), low_sat)
        mid = cv2.bitwise_and(
          cv2.inRange(gray, max(self.arrow_gray_min - 40, 95), white_lo - 1),
          low_sat,
        )
        body = cv2.bitwise_or(white, mid)
        # Remove fundo: so pixels com vizinho escuro (contorno preto / mapa).
        dark = cv2.inRange(gray, 0, 90)
        dark_near = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1)
        body = cv2.bitwise_and(body, dark_near)
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if int(np.count_nonzero(body)) < 12:
            return None

        white = cv2.bitwise_and(white, body)
        mid = cv2.bitwise_and(mid, body)
        wy, wx = np.where(white > 0)
        gy, gx = np.where(mid > 0)
        if wx.size < 4 or gx.size < 4:
            return None

        # Centroides em coords da imagem cheia.
        wcx = float(np.mean(wx)) + x0
        wcy = float(np.mean(wy)) + y0
        gcx = float(np.mean(gx)) + x0
        gcy = float(np.mean(gy)) + y0
        # Lateral: cinza → branco (branco a esquerda quando aponta pra cima).
        lx = wcx - gcx
        ly = wcy - gcy
        llen = math.hypot(lx, ly)
        if llen < 1.5:
            return None
        lx /= llen
        ly /= llen
        # Frente = 90° CCW do lateral.
        fx, fy = -ly, lx

        by, bx = np.where(body > 0)
        if bx.size < 8:
            return None
        pts_x = bx.astype(np.float64) + x0
        pts_y = by.astype(np.float64) + y0
        cx = float(np.mean(pts_x))
        cy = float(np.mean(pts_y))
        if math.hypot(cx - anchor_x, cy - anchor_y) > 20.0:
            return None

        # Nariz = extremo MAIS FINO (menor largura). Base tem asas largas.
        def _end_metrics(ux: float, uy: float) -> tuple[float, float, float, float]:
            proj = (pts_x - cx) * ux + (pts_y - cy) * uy
            tip_reach = float(np.max(proj))
            if tip_reach < 3.0:
                return 999.0, -1.0, cx, cy
            band = proj >= tip_reach * 0.45
            if int(np.count_nonzero(band)) < 2:
                return 999.0, tip_reach, cx, cy
            px_, py_ = -uy, ux
            lat = (pts_x[band] - cx) * px_ + (pts_y[band] - cy) * py_
            width = float(np.max(lat) - np.min(lat))
            tip_i = int(np.argmax(proj))
            return width, tip_reach, float(pts_x[tip_i]), float(pts_y[tip_i])

        w_a, r_a, tip_ax, tip_ay = _end_metrics(fx, fy)
        w_b, r_b, tip_bx, tip_by = _end_metrics(-fx, -fy)
        # Prefere extremo mais estreito; desempate por alcance.
        pick_a = (w_a, -r_a)
        pick_b = (w_b, -r_b)
        if pick_a <= pick_b and r_a > 0:
            angle = math.degrees(math.atan2(fy, fx))
            tip_x, tip_y = tip_ax, tip_ay
        elif r_b > 0:
            angle = math.degrees(math.atan2(-fy, -fx))
            tip_x, tip_y = tip_bx, tip_by
        else:
            return None

        tip_len = 10.0
        rad = math.radians(angle)
        return (
            anchor_x + math.cos(rad) * tip_len,
            anchor_y + math.sin(rad) * tip_len,
            angle,
        )

    def _detect_fixed_anchor(self, frame_bgr: np.ndarray) -> ArrowResult:
        if (
            self.fixed_player_anchor
            and not self.player_center_calibrated
            and self._auto_anchor_x is None
        ):
            ax, ay = self._estimate_player_anchor(frame_bgr)
            self._auto_anchor_x, self._auto_anchor_y = ax, ay

        anchor_x, anchor_y = self.player_position(frame_bgr.shape)
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        search = self._ui_exclusion_mask(width, height)
        search = cv2.bitwise_and(search, self._minimap_disc_mask(width, height))
        node_blobs = self._get_node_blobs(gray, hsv, search)

        def _commit(tip_x: float, tip_y: float, facing: float, *, contour=None) -> ArrowResult:
            # Evita flip 180° ao girar: escolhe entre facing e facing+180 o mais perto do anterior.
            if self._last_facing_deg is not None:
                cur = facing
                flipped = facing + 180.0
                def _delta(a: float, b: float) -> float:
                    d = abs((a - b + 180.0) % 360.0 - 180.0)
                    return d
                if _delta(flipped, self._last_facing_deg) + 25.0 < _delta(cur, self._last_facing_deg):
                    facing = flipped
                    rad = math.radians(facing)
                    tip_x = anchor_x + math.cos(rad) * 10.0
                    tip_y = anchor_y + math.sin(rad) * 10.0
            self._player_lost_frames = 0
            self._last_anchor_x = anchor_x
            self._last_anchor_y = anchor_y
            self._smooth_tip_x = tip_x
            self._smooth_tip_y = tip_y
            self._last_facing_deg = facing
            if contour is not None:
                self._last_arrow_contour = contour
            return ArrowResult(
                player_tracked=True,
                arrow_detected=True,
                player_x=anchor_x,
                player_y=anchor_y,
                arrow_tip_x=tip_x,
                arrow_tip_y=tip_y,
                arrow_angle_deg=facing,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )

        # 1) Bicolor: nariz = extremo mais fino da divisoria branco|cinza.
        bicolor = self._bicolor_tip_from_mask(frame_bgr, anchor_x, anchor_y)
        if bicolor is not None:
            tip_x, tip_y, facing = bicolor
            return _commit(tip_x, tip_y, facing)

        # 2) Template da seta de referencia (nariz = topo do asset).
        found = self._find_facing_at_fixed_anchor(
            gray, hsv, search, node_blobs, anchor_x, anchor_y
        )
        if found is not None:
            tip_x, tip_y, arrow_angle, _ax, _ay, contour = found
            facing = self._facing_from_tip(tip_x, tip_y, anchor_x, anchor_y)
            if facing is None:
                facing = arrow_angle
            if facing is not None:
                tip_len = 10.0
                rad = math.radians(facing)
                tip_x = anchor_x + math.cos(rad) * tip_len
                tip_y = anchor_y + math.sin(rad) * tip_len
                return _commit(tip_x, tip_y, facing, contour=contour)

        # 3) Radial ultimo recurso.
        radial = self._radial_fallback(
            frame_bgr, anchor_x, anchor_y, node_blobs=node_blobs
        )
        if radial is not None:
            tip_x, tip_y, arrow_angle = radial
            facing = self._facing_from_tip(tip_x, tip_y, anchor_x, anchor_y)
            if facing is None:
                facing = arrow_angle
            if facing is not None:
                tip_len = 10.0
                rad = math.radians(facing)
                tip_x = anchor_x + math.cos(rad) * tip_len
                tip_y = anchor_y + math.sin(rad) * tip_len
                return _commit(tip_x, tip_y, facing)

        if self._smooth_tip_x is not None and self._player_lost_frames < self._player_max_lost_frames:
            self._player_lost_frames += 1
            facing = self._facing_from_tip(
                self._smooth_tip_x, self._smooth_tip_y, anchor_x, anchor_y
            )
            if facing is not None:
                self._last_facing_deg = facing
            return ArrowResult(
                player_tracked=True,
                arrow_detected=False,
                player_x=anchor_x,
                player_y=anchor_y,
                arrow_tip_x=self._smooth_tip_x,
                arrow_tip_y=self._smooth_tip_y,
                arrow_angle_deg=facing or self._last_facing_deg,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )

        if self._last_facing_deg is not None and self._player_lost_frames < self._player_max_lost_frames * 2:
            self._player_lost_frames += 1
            return ArrowResult(
                player_tracked=True,
                arrow_detected=False,
                player_x=anchor_x,
                player_y=anchor_y,
                arrow_tip_x=self._smooth_tip_x,
                arrow_tip_y=self._smooth_tip_y,
                arrow_angle_deg=self._last_facing_deg,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )

        return ArrowResult(
            player_tracked=False,
            arrow_detected=False,
            player_x=anchor_x,
            player_y=anchor_y,
            arrow_angle_deg=self._last_facing_deg,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )

    def detect(self, frame_bgr: np.ndarray) -> ArrowResult:
        if self.fixed_player_anchor:
            return self._detect_fixed_anchor(frame_bgr)

        nominal_x, nominal_y = self.player_position(frame_bgr.shape)
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        search = self._ui_exclusion_mask(width, height)
        search = cv2.bitwise_and(search, self._minimap_disc_mask(width, height))
        node_blobs = self._get_node_blobs(gray, hsv, search)

        def _keep_valid(
            candidate: tuple[float, float, float, float, float, np.ndarray] | None,
        ) -> tuple[float, float, float, float, float, np.ndarray] | None:
            if candidate is None:
                return None
            tip_x, tip_y, _angle, anchor_x, anchor_y, contour = candidate
            if not self._validate_arrow_anchor(
                gray, hsv, anchor_x, anchor_y, node_blobs, tip_x=tip_x, tip_y=tip_y
            ):
                return None
            return candidate

        found = None
        if self._last_anchor_x is not None and self._last_anchor_y is not None:
            found = _keep_valid(self._find_arrow_local(gray, hsv, search, node_blobs))
        if found is None and self._last_anchor_x is not None:
            found = _keep_valid(self._find_arrow_near_last(gray, hsv, search, node_blobs))
        if found is None:
            found = _keep_valid(self._find_arrow_near_hint(gray, hsv, search, node_blobs))
        if found is None:
            hx = width * self.player_center_ratio[0]
            hy = height * self.player_center_ratio[1]
            found = _keep_valid(
                self._find_arrow_grid_near(
                    gray,
                    hsv,
                    search,
                    node_blobs,
                    hx,
                    hy,
                    max(self.arrow_search_radius_px, 55),
                )
            )
        if found is None:
            found = _keep_valid(
                self._find_arrow_by_template(gray, hsv, search, node_blobs, min_match=0.38)
            )

        if found is not None:
            tip_x, tip_y, arrow_angle, anchor_x, anchor_y, contour = found
            accepted = self._accept_tip(tip_x, tip_y, anchor_x, anchor_y)
            if accepted is None:
                radial = self._radial_fallback(frame_bgr, anchor_x, anchor_y)
                if radial is not None:
                    tip_x, tip_y, arrow_angle = radial
                    accepted = (tip_x, tip_y)
            if accepted is not None:
                sx, sy = accepted
                self._player_lost_frames = 0
                if contour is not None:
                    self._last_arrow_contour = contour
                self._last_anchor_x = anchor_x
                self._last_anchor_y = anchor_y
                self._smooth_tip_x = sx
                self._smooth_tip_y = sy
                facing = self._facing_from_tip(sx, sy, anchor_x, anchor_y)
                if facing is None and arrow_angle is not None:
                    facing = arrow_angle
                if facing is not None:
                    self._last_facing_deg = facing
                    return ArrowResult(
                        player_tracked=True,
                        arrow_detected=True,
                        player_x=anchor_x,
                        player_y=anchor_y,
                        arrow_tip_x=sx,
                        arrow_tip_y=sy,
                        arrow_angle_deg=facing,
                        anchor_x=anchor_x,
                        anchor_y=anchor_y,
                    )

        if self._smooth_tip_x is not None and self._player_lost_frames < self._player_max_lost_frames:
            self._player_lost_frames += 1
            ax = self._last_anchor_x if self._last_anchor_x is not None else nominal_x
            ay = self._last_anchor_y if self._last_anchor_y is not None else nominal_y
            facing = self._facing_from_tip(
                self._smooth_tip_x, self._smooth_tip_y, ax, ay
            )
            if facing is not None:
                self._last_facing_deg = facing
            return ArrowResult(
                player_tracked=True,
                arrow_detected=False,
                player_x=ax,
                player_y=ay,
                arrow_tip_x=self._smooth_tip_x,
                arrow_tip_y=self._smooth_tip_y,
                arrow_angle_deg=facing,
                anchor_x=ax,
                anchor_y=ay,
            )

        if self._last_facing_deg is not None and self._player_lost_frames < self._player_max_lost_frames * 2:
            self._player_lost_frames += 1
            if self._smooth_tip_x is not None and self._smooth_tip_y is not None:
                ax = self._last_anchor_x if self._last_anchor_x is not None else nominal_x
                ay = self._last_anchor_y if self._last_anchor_y is not None else nominal_y
                return ArrowResult(
                    player_tracked=True,
                    arrow_detected=False,
                    player_x=ax,
                    player_y=ay,
                    arrow_tip_x=self._smooth_tip_x,
                    arrow_tip_y=self._smooth_tip_y,
                    arrow_angle_deg=self._last_facing_deg,
                    anchor_x=ax,
                    anchor_y=ay,
                )

        self._smooth_tip_x = None
        self._smooth_tip_y = None
        self._last_arrow_contour = None
        self._player_lost_frames = 0
        return ArrowResult(
            player_tracked=False,
            arrow_detected=False,
            player_x=nominal_x,
            player_y=nominal_y,
            arrow_angle_deg=self._last_facing_deg,
        )

    def build_exclusion_mask(
        self,
        frame_shape: tuple[int, ...],
        result: ArrowResult,
        *,
        center_radius_px: float = 28.0,
        wedge_extra_px: float = 12.0,
    ) -> np.ndarray:
        """Mascara 255=manter, 0=excluir (seta do jogador + area central)."""
        height, width = frame_shape[:2]
        keep = np.full((height, width), 255, dtype=np.uint8)
        cx = int(result.pivot()[0])
        cy = int(result.pivot()[1])
        cv2.circle(keep, (cx, cy), max(int(center_radius_px), 1), 0, -1)

        if self._last_arrow_contour is not None:
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [self._last_arrow_contour], -1, 255, -1)
            contour_mask = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=1)
            keep = cv2.bitwise_and(keep, cv2.bitwise_not(contour_mask))

        if result.arrow_tip_x is not None and result.arrow_tip_y is not None:
            pivot_x, pivot_y = result.pivot()
            tip_x = float(result.arrow_tip_x)
            tip_y = float(result.arrow_tip_y)
            dx = tip_x - pivot_x
            dy = tip_y - pivot_y
            length = math.hypot(dx, dy)
            if length > 2.0:
                ux, uy = dx / length, dy / length
                pxn, pyn = -uy, ux
                base_w = 18.0
                tip_w = 12.0
                extend = wedge_extra_px
                pts = np.array(
                    [
                        [pivot_x - pxn * base_w, pivot_y - pyn * base_w],
                        [pivot_x + pxn * base_w, pivot_y + pyn * base_w],
                        [tip_x + pxn * tip_w + ux * extend, tip_y + pyn * tip_w + uy * extend],
                        [tip_x - pxn * tip_w + ux * extend, tip_y - pyn * tip_w + uy * extend],
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(keep, pts, 0)

        return keep
