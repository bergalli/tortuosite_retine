"""
main.py

Pipeline principal pour analyser la tortuosité de vaisseaux à partir d'une image.
Usage :
    python main.py
    python main.py --image /chemin/image.png --output results.csv
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from data_load import (
    build_fundus_mask,
    load_image,
    extract_green_channel,
    normalize_to_uint8,
    preprocess_green_channel,
)
from analyse import (
    detect_vessels,
    binarize_vessels,
    clean_binary_mask,
    skeletonize_mask,
    analyze_skeleton_tortuosity,
    select_first_significant_branches,
)


def save_intermediate_image(
    image: np.ndarray,
    output_path: Path,
    dilate_kernel_size: int = 0,
) -> None:
    """
    Sauvegarde une image intermédiaire dans un format lisible :
    - bool -> noir/blanc
    - float -> normalisation [0, 255]
    - RGB -> conversion vers BGR pour OpenCV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype == np.bool_:
        image_to_save = image.astype(np.uint8) * 255
    elif np.issubdtype(image.dtype, np.floating):
        image_to_save = normalize_to_uint8(image)
    elif image.dtype != np.uint8:
        image_to_save = normalize_to_uint8(image)
    else:
        image_to_save = image

    if dilate_kernel_size > 1:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
        image_to_save = cv2.dilate(image_to_save, kernel, iterations=1)

    if image_to_save.ndim == 3 and image_to_save.shape[2] == 3:
        image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), image_to_save)


def get_intermediate_dir(output_csv: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    return output_csv.parent / "output"


def create_skeleton_overlay(image: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """
    Superpose le squelette en rouge sur l'image d'origine pour un debug lisible.
    """
    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1)
    else:
        base = image.copy()

    if base.dtype != np.uint8:
        base = normalize_to_uint8(base)

    skeleton_mask = cv2.dilate(
        skeleton.astype(np.uint8) * 255,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    overlay = base.copy()
    overlay[skeleton_mask] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def run_pipeline(
    image_path: str,
    output_csv: str,
    max_branches: int = 3,
    output_dir: str | None = None,
    vessel_percentile: float = 95.0,
    vessel_low_percentile: float | None = 90.0,
):
    image_path = Path(image_path)
    output_csv = Path(output_csv)
    intermediate_dir = get_intermediate_dir(output_csv, output_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # 1. Charger l'image
    image = load_image(image_path)
    save_intermediate_image(image, intermediate_dir / "01_loaded_image.png")

    # 2. Extraire canal vert
    green = extract_green_channel(image)
    save_intermediate_image(green, intermediate_dir / "02_green_channel.png")
    fundus_mask = build_fundus_mask(green)
    save_intermediate_image(fundus_mask, intermediate_dir / "02b_fundus_mask.png")

    # 3. Prétraitement : contraste + flou
    preprocessed = preprocess_green_channel(green)
    save_intermediate_image(preprocessed, intermediate_dir / "03_preprocessed.png")

    # 4. Détection des vaisseaux
    vesselness = detect_vessels(preprocessed, mask=fundus_mask)
    save_intermediate_image(vesselness, intermediate_dir / "04_vesselness.png")

    # 5. Binarisation + nettoyage
    binary = binarize_vessels(
        vesselness,
        mask=fundus_mask,
        threshold_percentile=vessel_percentile,
        low_threshold_percentile=vessel_low_percentile,
    )
    save_intermediate_image(binary, intermediate_dir / "05_binary_mask.png")
    cleaned_binary = clean_binary_mask(binary, mask=fundus_mask)
    save_intermediate_image(cleaned_binary, intermediate_dir / "06_cleaned_mask.png")

    # 6. Skeletonisation
    skeleton = skeletonize_mask(cleaned_binary)
    save_intermediate_image(skeleton, intermediate_dir / "07_skeleton.png", dilate_kernel_size=3)
    save_intermediate_image(
        create_skeleton_overlay(image, skeleton),
        intermediate_dir / "07b_skeleton_overlay.png",
    )

    # 7. Analyse squelette + tortuosité
    summary = analyze_skeleton_tortuosity(skeleton)
    summary.to_csv(intermediate_dir / "08_full_skeleton_summary.csv", index=False)

    # 8. Garder les 3 premières branches significatives
    selected = select_first_significant_branches(
        summary,
        max_branches=max_branches,
        tortuosity_threshold=1.05,
    )

    # 9. Export CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_csv, index=False)

    print(f"Résultats sauvegardés dans : {output_csv}")
    print(f"Étapes intermédiaires sauvegardées dans : {intermediate_dir}")
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="demo/test.png",
        help="Chemin vers l'image PNG/TIF/JPG (défaut : demo/test.png)",
    )
    parser.add_argument(
        "--output",
        default="demo/results.csv",
        help="Chemin du CSV de sortie (défaut : demo/results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier des images intermédiaires (défaut : dossier output à côté du CSV)",
    )
    parser.add_argument("--max-branches", type=int, default=3, help="Nombre de branches à conserver")
    parser.add_argument(
        "--vessel-percentile",
        type=float,
        default=95.0,
        help=(
            "Seuil haut percentile sur la vesselness : plus bas = plus de vaisseaux "
            "(défaut : 95)"
        ),
    )
    parser.add_argument(
        "--vessel-low-percentile",
        type=float,
        default=90.0,
        help=(
            "Seuil bas percentile pour l'hystérésis : plus bas = extensions plus longues "
            "(défaut : 90 ; mettre >= --vessel-percentile pour désactiver)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.image,
        args.output,
        args.max_branches,
        args.output_dir,
        args.vessel_percentile,
        args.vessel_low_percentile,
    )
