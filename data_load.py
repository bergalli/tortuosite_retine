"""
data_load.py

Fonctions de chargement d'image et de prétraitement.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_holes


def load_image(path: str | Path) -> np.ndarray:
    """
    Charge une image avec OpenCV.

    OpenCV lit les images couleur en BGR, donc on convertit en RGB
    pour que le canal vert soit bien image[:, :, 1].
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image introuvable : {path}")

    image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {path}")

    # Image en niveaux de gris
    if image_bgr.ndim == 2:
        return image_bgr

    # Image couleur BGR ou BGRA
    if image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """
    Extrait le canal vert.
    Si l'image est déjà en niveaux de gris, on la retourne telle quelle.
    """
    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Image attendue : niveaux de gris ou RGB.")

    return image[:, :, 1]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convertit proprement une image en uint8 [0, 255].
    Utile si l'image est en 16-bit ou float.
    """
    image = image.astype(np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    image = image * 255

    return image.astype(np.uint8)


def preprocess_green_channel(green: np.ndarray) -> np.ndarray:
    """
    Équivalent approximatif de :
    - run("8-bit")
    - run("Enhance Contrast")
    - run("Gaussian Blur...", "sigma=1")

    Ici :
    - conversion uint8
    - CLAHE OpenCV pour améliorer le contraste local
    - flou gaussien sigma=1
    """
    green_uint8 = normalize_to_uint8(green)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(green_uint8)

    blurred = cv2.GaussianBlur(
        enhanced,
        ksize=(0, 0),
        sigmaX=1,
        sigmaY=1,
    )

    return blurred


def build_fundus_mask(
    green: np.ndarray,
    threshold: int = 8,
    max_threshold: int = 245,
    min_component_fraction: float = 0.05,
    hole_area_fraction: float = 0.003,
    erosion_fraction: float = 0.032,
    edge_margin_fraction: float = 0.014,
) -> np.ndarray:
    """
    Construit un masque du champ rétinien pour ignorer :
    - le fond noir
    - la séparation entre deux images
    - le bord circulaire du fond d'oeil
    """
    green_uint8 = normalize_to_uint8(green)
    height, width = green_uint8.shape

    # On exclut les pixels quasi blancs du cadre/séparateur avant de remplir
    # les disques rétiniens, sinon ils connectent toute l'image.
    mask = (green_uint8 > threshold) & (green_uint8 < max_threshold)

    morph_kernel_size = max(5, int(round(min(height, width) * 0.01)))
    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1
    morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)

    mask_uint8 = mask.astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, morph_kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, morph_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    min_component_area = int(height * width * min_component_fraction)

    filtered_mask = np.zeros_like(mask_uint8)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_component_area:
            filtered_mask[labels == label_idx] = 255

    mask = filtered_mask > 0
    hole_area = int(height * width * hole_area_fraction)
    mask = remove_small_holes(mask, max_size=hole_area)

    erosion_kernel_size = max(5, int(round(min(height, width) * erosion_fraction)))
    if erosion_kernel_size % 2 == 0:
        erosion_kernel_size += 1
    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=np.uint8)

    mask = cv2.erode(mask.astype(np.uint8) * 255, erosion_kernel, iterations=1) > 0

    edge_margin = max(1, int(round(min(height, width) * edge_margin_fraction)))
    mask[:edge_margin, :] = False
    mask[-edge_margin:, :] = False
    mask[:, :edge_margin] = False
    mask[:, -edge_margin:] = False

    return mask
