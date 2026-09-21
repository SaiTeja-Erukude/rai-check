from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

ImageTransform = Callable[[np.ndarray], np.ndarray]
Predictor = Callable[[np.ndarray], np.ndarray]


def adjust_brightness(images: np.ndarray, factor: float = 0.75) -> np.ndarray:
    """Scale image brightness while preserving the source dtype and value range."""
    return _restore_dtype(np.asarray(images), np.asarray(images, dtype=float) * factor)


def adjust_contrast(images: np.ndarray, factor: float = 0.65) -> np.ndarray:
    """Scale contrast around each image's mean intensity."""
    source = np.asarray(images)
    values = source.astype(float)
    axes = tuple(range(1, values.ndim))
    center = values.mean(axis=axes, keepdims=True)
    return _restore_dtype(source, center + factor * (values - center))


def horizontal_flip(images: np.ndarray) -> np.ndarray:
    """Flip a channels-last image batch across its width axis."""
    values = np.asarray(images)
    if values.ndim < 3:
        raise ValueError("images must have shape (batch, height, width[, channels])")
    width_axis = -2 if values.ndim >= 4 else -1
    return np.flip(values, axis=width_axis).copy()


def gaussian_noise(
    images: np.ndarray,
    stddev: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Add deterministic Gaussian noise relative to the image value range."""
    source = np.asarray(images)
    scale = _value_range(source)
    rng = np.random.default_rng(seed)
    noisy = source.astype(float) + rng.normal(0.0, stddev * scale, source.shape)
    return _restore_dtype(source, noisy)


DEFAULT_TRANSFORMATIONS: dict[str, ImageTransform] = {
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
    "horizontal_flip": horizontal_flip,
    "gaussian_noise": gaussian_noise,
}


def prediction_labels(predictions: np.ndarray) -> np.ndarray:
    """Convert class labels or a class-probability matrix into class labels."""
    values = np.asarray(predictions)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values.argmax(axis=1)
    raise ValueError("predictor output must be class labels or a 2D class-probability matrix")


def evaluate_transformations(
    predictor: Predictor,
    images: np.ndarray,
    transformations: Mapping[str, ImageTransform] | None = None,
) -> dict[str, np.ndarray]:
    """Run a predictor over transformed image batches and return class labels."""
    values = np.asarray(images)
    if values.ndim < 3:
        raise ValueError("images must have shape (batch, height, width[, channels])")
    selected = transformations or DEFAULT_TRANSFORMATIONS
    return {
        name: prediction_labels(predictor(transform(values)))
        for name, transform in selected.items()
    }


def _restore_dtype(source: np.ndarray, values: np.ndarray) -> np.ndarray:
    low, high = _clip_bounds(source)
    clipped = np.clip(values, low, high)
    return clipped.astype(source.dtype)


def _clip_bounds(images: np.ndarray) -> tuple[float, float]:
    if np.issubdtype(images.dtype, np.integer):
        info = np.iinfo(images.dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _value_range(images: np.ndarray) -> float:
    low, high = _clip_bounds(images)
    return high - low
