from __future__ import annotations

import numpy as np


class SceneDetector:
    def __init__(self, threshold: float = 0.32, sample_step: int = 8) -> None:
        self.threshold = threshold
        self.sample_step = max(1, sample_step)

    def score(self, first: np.ndarray, second: np.ndarray) -> float:
        a = first[:: self.sample_step, :: self.sample_step].astype(np.float32)
        b = second[:: self.sample_step, :: self.sample_step].astype(np.float32)
        # Luma is less sensitive than RGB to isolated chroma noise.
        luma_a = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
        luma_b = b[..., 0] * 0.2126 + b[..., 1] * 0.7152 + b[..., 2] * 0.0722
        return float(np.mean(np.abs(luma_a - luma_b)) / 255.0)

    def is_cut(self, first: np.ndarray, second: np.ndarray) -> bool:
        return self.score(first, second) >= self.threshold
