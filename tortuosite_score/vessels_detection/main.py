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

from tortuosite_score.vessels_detection.data_load import (
    build_fundus_mask,
    load_image,
    extract_green_channel,
    normalize_to_uint8,
    preprocess_green_channel,
)
from tortuosite_score.vessels_detection.analyse import (
    detect_vessels,
    binarize_vessels,
    clean_binary_mask,
    skeletonize_mask,
    analyze_skeleton_tortuosity,
    select_first_significant_branches,
)
from tortuosite_score.vessels_detection.dl_extra import (
    deep_learning_available,
    require_deep_learning,
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


def create_vascx_av_overlay(image: np.ndarray, artery_vein_classes: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        base = normalize_to_uint8(image)
    else:
        base = image.copy()

    color_mask = np.zeros_like(base)
    color_mask[artery_vein_classes == 1] = np.array([255, 0, 0], dtype=np.uint8)
    color_mask[artery_vein_classes == 2] = np.array([0, 80, 255], dtype=np.uint8)
    color_mask[artery_vein_classes == 3] = np.array([0, 255, 0], dtype=np.uint8)
    mask = artery_vein_classes > 0
    overlay = base.copy()
    overlay[mask] = (0.45 * base[mask] + 0.55 * color_mask[mask]).astype(np.uint8)
    return overlay


def create_disc_overlay(image: np.ndarray, disc_mask: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        base = normalize_to_uint8(image)
    else:
        base = image.copy()

    overlay = base.copy()
    overlay[disc_mask] = (0.45 * base[disc_mask] + 0.55 * np.array([255, 255, 0])).astype(
        np.uint8
    )
    return overlay


def create_optic_disc_root_overlay(
    image: np.ndarray,
    skeleton: np.ndarray,
    disc_mask: np.ndarray,
) -> np.ndarray:
    overlay = create_skeleton_overlay(create_disc_overlay(image, disc_mask), skeleton)
    if not np.any(disc_mask):
        return overlay

    rows, cols = np.where(disc_mask)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    radius = int(max(y_max - y_min + 1, x_max - x_min + 1) * 1.6)
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    y0 = max(0, center_y - radius)
    y1 = min(overlay.shape[0], center_y + radius)
    x0 = max(0, center_x - radius)
    x1 = min(overlay.shape[1], center_x + radius)
    return overlay[y0:y1, x0:x1]


def estimate_optic_disc_mask(
    image: np.ndarray,
    vessel_mask: np.ndarray,
    fundus_mask: np.ndarray,
) -> np.ndarray:
    """
    Fallback for UWF images where the VascX disc model returns no disc.

    This is used only for debug crops/overlays. It does not alter vessel
    segmentation or tortuosity scoring.
    """
    if image.dtype != np.uint8:
        image_uint8 = normalize_to_uint8(image)
    else:
        image_uint8 = image

    red = image_uint8[:, :, 0].astype(np.float32)
    green = image_uint8[:, :, 1].astype(np.float32)
    blue = image_uint8[:, :, 2].astype(np.float32)
    brightness = 0.45 * red + 0.45 * green - 0.10 * blue

    if not np.any(fundus_mask):
        return np.zeros(fundus_mask.shape, dtype=bool)

    threshold = np.percentile(brightness[fundus_mask], 99.2)
    bright = (brightness >= threshold) & fundus_mask
    bright = cv2.morphologyEx(
        bright.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros(fundus_mask.shape, dtype=bool)

    vessel_density = cv2.GaussianBlur(
        vessel_mask.astype(np.float32),
        (0, 0),
        sigmaX=28,
        sigmaY=28,
    )
    best_label = 0
    best_score = -1.0
    for label in range(1, component_count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        cx, cy = centroids[label]
        y = int(np.clip(round(cy), 0, vessel_density.shape[0] - 1))
        x = int(np.clip(round(cx), 0, vessel_density.shape[1] - 1))
        score = float(vessel_density[y, x]) * np.sqrt(area)
        if score > best_score:
            best_label = label
            best_score = score

    if best_label == 0:
        return np.zeros(fundus_mask.shape, dtype=bool)

    disc = labels == best_label
    disc = cv2.dilate(
        disc.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
        iterations=1,
    ).astype(bool)
    return disc & fundus_mask


def run_pipeline(
    image_path: str,
    output_csv: str,
    max_branches: int = 3,
    output_dir: str | None = None,
    method: str = "deep",
    vessel_percentile: float = 95.0,
    vessel_low_percentile: float | None = 90.0,
    deep_threshold: float = 0.30,
    deep_modality: str = "CFP",
    deep_backend: str = "DCP",
    vascx_av_size: int = 1024,
    vascx_use_contrast_enhancement: bool = True,
    vascx_min_object_size: int = 12,
    vascx_closing_radius: int = 1,
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

    # 2bis. Masque du champ rétinien
    fundus_mask = build_fundus_mask(green)
    save_intermediate_image(fundus_mask, intermediate_dir / "02b_fundus_mask.png")

    # 3. Prétraitement classique
    preprocessed = preprocess_green_channel(green)
    save_intermediate_image(preprocessed, intermediate_dir / "03_preprocessed.png")

    # 4. Segmentation des vaisseaux
    artery_binary = None
    vein_binary = None
    disc_mask = None
    if method == "deep":
        require_deep_learning()
        from tortuosite_score.vessels_detection.deep_model import predict_vessels_deep
        from tortuosite_score.vessels_detection.vascx_model import predict_vascx

        normalized_backend = deep_backend.strip().upper()
        if normalized_backend == "DCP":
            print("Méthode utilisée : deep learning DCP")
            print(f"Modalité DCP : {deep_modality}")

            vessel_prob = predict_vessels_deep(
                image,
                mask=fundus_mask,
                modality=deep_modality,
            )

            save_intermediate_image(
                vessel_prob,
                intermediate_dir / "04_deep_vessel_probability.png",
            )

            binary = vessel_prob > deep_threshold
            binary &= fundus_mask

            save_intermediate_image(
                binary,
                intermediate_dir / "05_binary_mask.png",
            )
        elif normalized_backend == "VASCX":
            print("Méthode utilisée : deep learning VascX")
            print(
                "[VascX] paramètres "
                f"av_size={vascx_av_size} "
                f"use_contrast_enhancement={vascx_use_contrast_enhancement} "
                f"min_object_size={vascx_min_object_size} "
                f"closing_radius={vascx_closing_radius}"
            )
            vascx_prediction = predict_vascx(
                image,
                mask=fundus_mask,
                av_size=vascx_av_size,
                use_contrast_enhancement=vascx_use_contrast_enhancement,
            )

            binary = vascx_prediction.vessel_mask
            artery_binary = vascx_prediction.artery_mask
            vein_binary = vascx_prediction.vein_mask
            disc_mask = vascx_prediction.disc_mask

            save_intermediate_image(
                vascx_prediction.vessel_mask,
                intermediate_dir / "04_vascx_vessel_mask.png",
            )
            save_intermediate_image(
                create_vascx_av_overlay(image, vascx_prediction.artery_vein_classes),
                intermediate_dir / "04b_vascx_artery_vein_overlay.png",
            )
            save_intermediate_image(
                vascx_prediction.disc_mask,
                intermediate_dir / "04c_vascx_disc_mask.png",
            )
            save_intermediate_image(
                create_disc_overlay(image, vascx_prediction.disc_mask),
                intermediate_dir / "04d_vascx_disc_overlay.png",
            )
            if not np.any(disc_mask):
                disc_mask = estimate_optic_disc_mask(
                    image,
                    vessel_mask=vascx_prediction.vessel_mask,
                    fundus_mask=fundus_mask,
                )
                if np.any(disc_mask):
                    print("[VascX] disc model returned empty mask; using debug-only disc estimate.")
                    save_intermediate_image(
                        disc_mask,
                        intermediate_dir / "04e_estimated_disc_mask.png",
                    )
                    save_intermediate_image(
                        create_disc_overlay(image, disc_mask),
                        intermediate_dir / "04f_estimated_disc_overlay.png",
                    )
            save_intermediate_image(
                binary,
                intermediate_dir / "05_binary_mask.png",
            )
        else:
            raise ValueError(
                f"Deep backend inconnu : {deep_backend}. Utilise 'DCP' ou 'VascX'."
            )

    elif method == "classical":
        print("Méthode utilisée : classique Frangi")

        vesselness = detect_vessels(
            preprocessed,
            mask=fundus_mask,
        )

        save_intermediate_image(
            vesselness,
            intermediate_dir / "04_vesselness.png",
        )

        binary = binarize_vessels(
            vesselness,
            mask=fundus_mask,
            threshold_percentile=vessel_percentile,
            low_threshold_percentile=vessel_low_percentile,
        )

        save_intermediate_image(
            binary,
            intermediate_dir / "05_binary_mask.png",
        )

    else:
        raise ValueError(
            f"Méthode inconnue : {method}. Utilise 'deep' ou 'classical'."
        )


    # 5. Nettoyage
    if artery_binary is not None and vein_binary is not None:
        cleaned_artery = clean_binary_mask(
            artery_binary,
            mask=fundus_mask,
            min_object_size=vascx_min_object_size,
            closing_radius=vascx_closing_radius,
        )
        cleaned_vein = clean_binary_mask(
            vein_binary,
            mask=fundus_mask,
            min_object_size=vascx_min_object_size,
            closing_radius=vascx_closing_radius,
        )
        save_intermediate_image(
            cleaned_artery,
            intermediate_dir / "06_cleaned_artery_mask.png",
        )
        save_intermediate_image(
            cleaned_vein,
            intermediate_dir / "06_cleaned_vein_mask.png",
        )
        cleaned_binary = cleaned_artery | cleaned_vein
    else:
        cleaned_binary = clean_binary_mask(
            binary,
            mask=fundus_mask,
        )
    save_intermediate_image(
        cleaned_binary,
        intermediate_dir / "06_cleaned_mask.png",
    )

    # 6. Skeletonisation
    if artery_binary is not None and vein_binary is not None:
        skeleton = skeletonize_mask(cleaned_artery) | skeletonize_mask(cleaned_vein)
    else:
        skeleton = skeletonize_mask(cleaned_binary)

    save_intermediate_image(
        skeleton,
        intermediate_dir / "07_skeleton.png",
        dilate_kernel_size=3,
    )

    save_intermediate_image(
        create_skeleton_overlay(image, skeleton),
        intermediate_dir / "07b_skeleton_overlay.png",
    )
    if disc_mask is not None:
        save_intermediate_image(
            create_optic_disc_root_overlay(image, skeleton, disc_mask),
            intermediate_dir / "07c_optic_disc_root_overlay.png",
        )

    # 7. Analyse squelette + tortuosité
    summary = analyze_skeleton_tortuosity(skeleton)
    summary.to_csv(
        intermediate_dir / "08_full_skeleton_summary.csv",
        index=False,
    )

    # 8. Garder les premières branches significatives
    selected = select_first_significant_branches(
        summary,
        max_branches=max_branches,
        tortuosity_threshold=1.05,
    )

    # 9. Export CSV final
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
    )

    parser.add_argument(
        "--output",
        default="demo/results.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--method",
        choices=["deep", "classical"],
        default="deep" if deep_learning_available() else "classical",
    )

    parser.add_argument(
        "--deep-threshold",
        type=float,
        default=0.30,
        help="Seuil de probabilité pour le modèle deep learning",
    )

    parser.add_argument(
        "--max-branches",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--vessel-percentile",
        type=float,
        default=95.0,
    )

    parser.add_argument(
        "--vessel-low-percentile",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--deep-modality",
        choices=["CFP", "UWF", "FFA", "SLO", "OCTA"],
        default="CFP",
        help="Modalité pour le modèle DCP : CFP ou UWF principalement.",
    )
    parser.add_argument(
        "--deep-backend",
        choices=["DCP", "VascX"],
        default="DCP",
        help="Backend deep learning : DCP historique ou VascX.",
    )
    parser.add_argument(
        "--vascx-av-size",
        type=int,
        choices=[512, 768, 1024, 1280],
        default=1024,
        help="Taille d'entrée VascX pour la segmentation artère/veine.",
    )
    parser.add_argument(
        "--no-vascx-contrast-enhancement",
        action="store_true",
        help="Désactive le rehaussement de contraste interne VascX.",
    )
    parser.add_argument(
        "--vascx-min-object-size",
        type=int,
        default=12,
        help="Taille minimale des objets gardés après segmentation VascX.",
    )
    parser.add_argument(
        "--vascx-closing-radius",
        type=int,
        default=1,
        help="Rayon de fermeture morphologique après segmentation VascX.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        image_path=args.image,
        output_csv=args.output,
        max_branches=args.max_branches,
        output_dir=args.output_dir,
        method=args.method,
        vessel_percentile=args.vessel_percentile,
        vessel_low_percentile=args.vessel_low_percentile,
        deep_threshold=args.deep_threshold,
        deep_modality=args.deep_modality,
        deep_backend=args.deep_backend,
        vascx_av_size=args.vascx_av_size,
        vascx_use_contrast_enhancement=not args.no_vascx_contrast_enhancement,
        vascx_min_object_size=args.vascx_min_object_size,
        vascx_closing_radius=args.vascx_closing_radius,
    )
