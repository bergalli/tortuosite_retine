"""
Wrapper for VascX vessel, artery/vein, and optic disc segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from tortuosite_score.vessels_detection.deep_model import _normalize_rgb_uint8


AV_MODEL_ID = "Eyened/vascx:artery_vein/av_july24.pt"
DISC_MODEL_ID = "Eyened/vascx:disc/disc_july24.pt"

_AV_MODEL = None
_DISC_MODEL = None


@dataclass(frozen=True)
class VascXPrediction:
    artery_vein_classes: np.ndarray
    disc_classes: np.ndarray
    vessel_mask: np.ndarray
    artery_mask: np.ndarray
    vein_mask: np.ndarray
    disc_mask: np.ndarray


def _select_device() -> tuple[str, bool]:
    try:
        import torch
    except Exception:
        return "cpu", False

    if torch.cuda.is_available():
        return "cuda", True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def _load_segmentation_model(model_id: str, size: int):
    try:
        from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
    except Exception as exc:
        raise RuntimeError(
            "VascX backend requires the optional package `vascx-simplify`. "
            "Install it with `uv add 'vascx-simplify>=0.1.11'` or use the DCP backend."
        ) from exc

    device, use_fp16 = _select_device()
    model_path = from_huggingface(model_id)
    transform = VASCXTransform(size=size, use_fp16=use_fp16, device=device)
    return EnsembleSegmentation(model_path, transform, device=device)


def _get_av_model():
    global _AV_MODEL
    if _AV_MODEL is None:
        _AV_MODEL = _load_segmentation_model(AV_MODEL_ID, size=1024)
    return _AV_MODEL


def _get_disc_model():
    global _DISC_MODEL
    if _DISC_MODEL is None:
        _DISC_MODEL = _load_segmentation_model(DISC_MODEL_ID, size=512)
    return _DISC_MODEL


def _prediction_to_classes(prediction, shape: tuple[int, int]) -> np.ndarray:
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    classes = np.asarray(prediction)
    if classes.ndim == 3:
        classes = classes[0]
    if classes.ndim != 2:
        raise RuntimeError(f"Unexpected VascX prediction shape: {classes.shape}")

    target_h, target_w = shape
    if classes.shape != (target_h, target_w):
        classes = cv2.resize(
            classes.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return classes.astype(np.uint8)


def predict_vascx(image_rgb: np.ndarray, mask: np.ndarray | None = None) -> VascXPrediction:
    """
    Run VascX artery/vein and optic disc segmentation.

    VascX artery/vein classes are:
    - 0: background
    - 1: artery
    - 2: vein
    - 3: crossing
    """
    image_rgb = _normalize_rgb_uint8(image_rgb)
    image_pil = Image.fromarray(image_rgb)
    shape = image_rgb.shape[:2]

    av_classes = _prediction_to_classes(_get_av_model().predict(image_pil), shape)
    disc_classes = _prediction_to_classes(_get_disc_model().predict(image_pil), shape)

    vessel_mask = av_classes > 0
    artery_mask = (av_classes == 1) | (av_classes == 3)
    vein_mask = (av_classes == 2) | (av_classes == 3)
    disc_mask = disc_classes > 0

    if mask is not None:
        vessel_mask &= mask
        artery_mask &= mask
        vein_mask &= mask
        disc_mask &= mask
        av_classes = np.where(mask, av_classes, 0).astype(np.uint8)
        disc_classes = np.where(mask, disc_classes, 0).astype(np.uint8)

    print(
        "[VascX] pixels "
        f"vessel={int(vessel_mask.sum())} "
        f"artery={int(artery_mask.sum())} "
        f"vein={int(vein_mask.sum())} "
        f"disc={int(disc_mask.sum())}"
    )

    return VascXPrediction(
        artery_vein_classes=av_classes,
        disc_classes=disc_classes,
        vessel_mask=vessel_mask,
        artery_mask=artery_mask,
        vein_mask=vein_mask,
        disc_mask=disc_mask,
    )
