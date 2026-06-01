"""
analyse.py

Fonctions de détection des vaisseaux, skeletonisation et calcul de tortuosité.
"""

import numpy as np
import pandas as pd

from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.morphology import (
    closing,
    disk,
    skeletonize,
    remove_small_objects,
    remove_small_holes,
)
from tortuosite_score.vessels_detection.skan_extra import skan_available
from tortuosite_score.vessels_detection.skeleton_graph import summarize_skeleton_branches


def detect_vessels(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Remplacement Python approximatif du plugin Fiji Ridge Detection.

    On utilise Frangi, souvent adapté aux structures tubulaires comme :
    - vaisseaux
    - neurites
    - fibres

    Paramètres à ajuster selon tes images :
    - sigmas : largeur attendue des vaisseaux
    - black_ridges=True : vaisseaux sombres sur fond plus clair
    """
    image_float = image.astype(np.float32) / 255.0

    vesselness = frangi(
        image_float,
        sigmas=range(1, 4),
        black_ridges=True,
    )

    if mask is not None:
        vesselness = np.where(mask, vesselness, 0.0)

    return vesselness


def binarize_vessels(
    vesselness: np.ndarray,
    mask: np.ndarray | None = None,
    threshold_percentile: float = 95.0,
    low_threshold_percentile: float | None = 90.0,
) -> np.ndarray:
    """
    Binarisation de la vesselness.

    L'hystérésis garde les réponses faibles seulement si elles sont connectées
    à une réponse forte. C'est plus adapté qu'un simple seuil bas pour récupérer
    les petits vaisseaux sans ouvrir trop fort au bruit.
    """
    if mask is not None and np.any(mask):
        values = vesselness[mask]
    else:
        values = vesselness

    high_threshold = np.percentile(values, threshold_percentile)
    if low_threshold_percentile is None or low_threshold_percentile >= threshold_percentile:
        binary = vesselness > high_threshold
    else:
        low_threshold = np.percentile(values, low_threshold_percentile)
        binary = apply_hysteresis_threshold(
            vesselness,
            low_threshold,
            high_threshold,
        )

    if mask is not None:
        binary &= mask

    return binary


def clean_binary_mask(
    binary: np.ndarray,
    mask: np.ndarray | None = None,
    min_object_size: int = 12,
    min_hole_size: int = 12,
    closing_radius: int = 1,
) -> np.ndarray:
    """
    Nettoyage du masque binaire.
    Équivalent approximatif de :
    - filtres morphologiques pour enlever le bruit

    On évite le median blur ici : sur des vaisseaux fins, il efface facilement
    des branches utiles avant la skeletonisation.
    """
    cleaned = binary

    cleaned = remove_small_objects(cleaned, max_size=max(0, min_object_size - 1))
    cleaned = remove_small_holes(cleaned, max_size=max(0, min_hole_size - 1))

    if closing_radius > 0:
        cleaned = closing(cleaned, footprint=disk(closing_radius))

    if mask is not None:
        cleaned &= mask

    return cleaned


def skeletonize_mask(binary: np.ndarray) -> np.ndarray:
    """
    Skeletonisation du masque.
    Équivalent de run("Skeletonize").
    """
    return skeletonize(binary)


def analyze_skeleton_tortuosity(skeleton: np.ndarray) -> pd.DataFrame:
    """
    Analyse du squelette (skan si installé, sinon graphe scikit-image).

    Cela remplace partiellement :
        run("Analyze Skeleton (2D/3D)", ...)

    Colonnes importantes :
    - branch-distance : longueur réelle le long du squelette
    - euclidean-distance : distance droite entre début et fin de branche
    """
    if skan_available():
        from skan import Skeleton, summarize

        skel = Skeleton(skeleton)
        summary = summarize(skel, separator="-")
    else:
        summary = summarize_skeleton_branches(skeleton)

    # Sécurité : éviter division par zéro
    summary = summary.copy()
    summary = summary[summary["euclidean-distance"] > 0]

    summary["tortuosity"] = (
        summary["branch-distance"] / summary["euclidean-distance"]
    )

    return summary


def select_first_significant_branches(
    summary: pd.DataFrame,
    max_branches: int = 3,
    tortuosity_threshold: float = 1.05,
) -> pd.DataFrame:
    """
    Sélectionne les premières branches significatives, comme dans ta macro Fiji.

    Critère :
    - longueur > 0
    - distance euclidienne > 0
    - tortuosité > 1.05
    """
    selected = summary[
        (summary["branch-distance"] > 0)
        & (summary["euclidean-distance"] > 0)
        & (summary["tortuosity"] > tortuosity_threshold)
    ].head(max_branches)

    # Colonnes simplifiées pour CSV final
    output = pd.DataFrame({
        "BranchID": selected.index,
        "Length": selected["branch-distance"].values,
        "Euclidean": selected["euclidean-distance"].values,
        "Tortuosity": selected["tortuosity"].values,
    })

    return output
