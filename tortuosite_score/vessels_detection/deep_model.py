"""
deep_model.py

Wrapper local pour le modèle DCP / UNet_DCP_1024 :
Broad-domain retinal vessel segmentation.

Le code :
- télécharge le repo officiel DCP depuis GitHub
- télécharge le checkpoint Hugging Face AIMClab-RUC/UNet_DCP_1024
- utilise l'inference officielle du repo
- retourne une carte de probabilité [0, 1] à la taille originale
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

from tortuosite_score.vessels_detection.dl_extra import DEEP_INSTALL_HINT, require_deep_learning
from tortuosite_score.vessels_detection.image_utils import normalize_rgb_uint8


DCP_REPO_ZIP_URL = "https://github.com/ruc-aimc-lab/dcp/archive/refs/heads/main.zip"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DCP_SOURCE_DIR = PACKAGE_ROOT / "external_models" / "dcp"
DCP_CHECKPOINT_DIR = PACKAGE_ROOT / "external_models" / "UNet_DCP_1024"

_DCP_ENGINE = None


def _download_dcp_source() -> Path:
    """
    Télécharge le code officiel DCP si absent.
    """
    if (DCP_SOURCE_DIR / "inference.py").exists() and (DCP_SOURCE_DIR / "models").exists():
        return DCP_SOURCE_DIR

    DCP_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)

    zip_path = DCP_SOURCE_DIR.parent / "dcp-main.zip"
    extract_dir = DCP_SOURCE_DIR.parent / "_dcp_extract"

    print("[DCP] Téléchargement du code officiel DCP...")
    print(f"[DCP] URL : {DCP_REPO_ZIP_URL}")

    try:
        urllib.request.urlretrieve(DCP_REPO_ZIP_URL, zip_path)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de télécharger le code DCP automatiquement.\n"
            "Alternative manuelle :\n"
            "  mkdir -p external_models\n"
            "  git clone https://github.com/ruc-aimc-lab/dcp external_models/dcp\n"
        ) from exc

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    extracted_root = extract_dir / "dcp-main"

    if not extracted_root.exists():
        raise RuntimeError("Extraction du repo DCP échouée.")

    if DCP_SOURCE_DIR.exists():
        shutil.rmtree(DCP_SOURCE_DIR)

    shutil.move(str(extracted_root), str(DCP_SOURCE_DIR))

    shutil.rmtree(extract_dir)
    zip_path.unlink(missing_ok=True)

    return DCP_SOURCE_DIR


def _download_dcp_checkpoint() -> Path:
    """
    Télécharge le checkpoint Hugging Face.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "DCP checkpoint download requires huggingface-hub. "
            f"Install with: {DEEP_INSTALL_HINT}"
        ) from exc

    print("[DCP] Vérification / téléchargement du checkpoint UNet_DCP_1024...")

    model_dir = snapshot_download(
        repo_id="AIMClab-RUC/UNet_DCP_1024",
        local_dir=str(DCP_CHECKPOINT_DIR),
    )

    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    model_path = model_dir / "model.pkl"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json introuvable dans {model_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl introuvable dans {model_dir}")

    return model_dir


def _get_dcp_engine():
    """
    Charge une seule fois le moteur d'inférence DCP.
    """
    global _DCP_ENGINE

    if _DCP_ENGINE is not None:
        return _DCP_ENGINE

    source_dir = _download_dcp_source()
    checkpoint_dir = _download_dcp_checkpoint()

    source_dir_abs = str(source_dir.resolve())

    if source_dir_abs not in sys.path:
        sys.path.insert(0, source_dir_abs)

    try:
        from inference import Inference
    except Exception as exc:
        raise RuntimeError(
            "Impossible d'importer l'inférence DCP.\n"
            "Vérifie que le dossier external_models/dcp contient bien inference.py et models/."
        ) from exc

    _DCP_ENGINE = Inference(model_path=str(checkpoint_dir))

    return _DCP_ENGINE


def _print_probability_stats(prob: np.ndarray, mask: np.ndarray | None = None) -> None:
    """
    Debug utile : si tout est plat, le modèle / preprocessing est encore mauvais.
    """
    if mask is not None and np.any(mask):
        values = prob[mask]
    else:
        values = prob.reshape(-1)

    if values.size == 0:
        print("[DCP] Warning : aucune valeur dans le masque.")
        return

    q = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])

    print(
        "[DCP] prob stats "
        f"min={q[0]:.4f} "
        f"q25={q[1]:.4f} "
        f"median={q[2]:.4f} "
        f"q75={q[3]:.4f} "
        f"q90={q[4]:.4f} "
        f"q95={q[5]:.4f} "
        f"q99={q[6]:.4f} "
        f"max={q[7]:.4f}"
    )


def predict_vessels_deep(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    modality: str = "CFP",
) -> np.ndarray:
    """
    Segmentation deep learning des vaisseaux.

    Parameters
    ----------
    image_rgb:
        Image RGB chargée par ton load_image().
    mask:
        Masque du fond d'oeil. Les pixels hors masque seront forcés à 0.
    modality:
        Modalité pour DCP :
        - "CFP" : fundus couleur classique
        - "UWF" : ultra-widefield
        - "FFA", "SLO", "OCTA" : autres modalités supportées

    Returns
    -------
    prob:
        Carte de probabilité float32 entre 0 et 1, taille originale.
    """
    valid_modalities = {"CFP", "UWF", "FFA", "SLO", "OCTA"}

    if modality not in valid_modalities:
        raise ValueError(
            f"Modalité inconnue : {modality}. "
            f"Modalités possibles : {sorted(valid_modalities)}"
        )

    require_deep_learning("DCP vessel segmentation")

    image_rgb = normalize_rgb_uint8(image_rgb)
    original_h, original_w = image_rgb.shape[:2]

    engine = _get_dcp_engine()

    # L'inference officielle accepte une image numpy RGB.
    # Elle resize selon config.json, applique /255, passe le modèle,
    # puis applique sigmoid.
    pred_uint8 = engine.inference(image_rgb, modality)

    prob = pred_uint8.astype(np.float32) / 255.0

    if prob.shape[:2] != (original_h, original_w):
        prob = cv2.resize(
            prob,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

    prob = np.clip(prob, 0.0, 1.0)

    if mask is not None:
        prob = np.where(mask, prob, 0.0)

    _print_probability_stats(prob, mask)

    return prob.astype(np.float32)
