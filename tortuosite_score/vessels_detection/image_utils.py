"""Image preprocessing helpers shared by vessel detection backends."""

from __future__ import annotations

import numpy as np


def normalize_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))

    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    image = np.clip(image * 255.0, 0, 255)

    return image.astype(np.uint8)
