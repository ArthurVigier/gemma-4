"""Vision pipeline that projects simulated observations onto ARC-like grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arc_types import ArcGrid
from .config import VisionConfig


DEFAULT_ARC_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 116, 217],
        [255, 65, 54],
        [46, 204, 64],
        [255, 220, 0],
        [170, 170, 170],
        [240, 18, 190],
        [255, 133, 27],
        [127, 219, 255],
        [135, 12, 37],
    ],
    dtype=np.uint8,
)


@dataclass(slots=True)
class VisionConversionResult:
    """Final grid and calibration metadata."""

    grid: ArcGrid
    palette: np.ndarray
    original_shape: tuple[int, int, int]


class VisionGridConverter:
    """Converts RGB observations into compact ARC-style symbolic grids."""

    def __init__(self, config: VisionConfig | None = None, palette: np.ndarray | None = None) -> None:
        self.config = config or VisionConfig()
        self.palette = DEFAULT_ARC_PALETTE if palette is None else np.asarray(palette, dtype=np.uint8)
        if self.palette.shape != (self.config.color_count, 3):
            raise ValueError("Palette must match the configured color count and contain RGB triplets.")

    def convert_rgb(self, image: np.ndarray) -> VisionConversionResult:
        """Resize via nearest-neighbor and quantize pixels to the ARC palette."""

        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3).")

        resized = self._nearest_resize(rgb, self.config.grid_height, self.config.grid_width)
        quantized = self._quantize_to_palette(resized)
        return VisionConversionResult(
            grid=ArcGrid(values=quantized.astype(np.int64)),
            palette=self.palette.copy(),
            original_shape=tuple(rgb.shape),
        )

    @staticmethod
    def _nearest_resize(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resizes with deterministic nearest-neighbor sampling and no extra dependencies."""

        source_h, source_w, _ = image.shape
        row_idx = np.linspace(0, source_h - 1, target_h).round().astype(int)
        col_idx = np.linspace(0, source_w - 1, target_w).round().astype(int)
        return image[row_idx][:, col_idx]

    def _quantize_to_palette(self, image: np.ndarray) -> np.ndarray:
        """Maps each RGB pixel to the nearest symbolic ARC color index."""

        flat = image.reshape(-1, 3).astype(np.int32)
        palette = self.palette.astype(np.int32)
        distances = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1).reshape(self.config.grid_height, self.config.grid_width)
