"""
deep_model.py

Inference deep learning pour segmentation de vaisseaux rétiniens.
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.middle = DoubleConv(1024, 2048)

        self.up4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.conv4 = DoubleConv(2048, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv3 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        mid = self.middle(self.pool(d4))

        x = self.up4(mid)
        x = torch.cat([x, d4], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.conv1(x)

        return self.out(x)

def _find_weight_file(model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    candidates = list(model_dir.rglob("*.pth")) + list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.bin"))

    if not candidates:
        raise FileNotFoundError(f"Aucun fichier de poids trouvé dans {model_dir}")

    return candidates[0]


def load_retina_unet(device: str | None = None) -> tuple[nn.Module, str]:
    """
    Télécharge et charge le modèle Hugging Face.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = snapshot_download(
        repo_id="IbrahimDayax/retina-vessel-unet-segmentation",
        cache_dir=str(Path.home() / ".cache" / "huggingface"),
    )

    weight_path = _find_weight_file(model_dir)

    model = UNet().to(device)
    state = torch.load(weight_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Nettoie les préfixes éventuels
    clean_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=False)
    model.eval()

    return model, device


def predict_vessels_deep(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    input_size: int = 512,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Retourne une probabilité de vaisseau entre 0 et 1.

    image_rgb : image RGB chargée par ton load_image()
    mask : fundus_mask ou valid_vessel_area
    threshold : laisse None pour retourner une probabilité continue
    """
    model, device = load_retina_unet()

    original_h, original_w = image_rgb.shape[:2]

    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.float32)
        image_rgb = image_rgb - image_rgb.min()
        image_rgb = image_rgb / max(image_rgb.max(), 1e-8)
        image_rgb = (image_rgb * 255).astype(np.uint8)

    resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)

    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    if mask is not None:
        prob = np.where(mask, prob, 0.0)

    if threshold is not None:
        return prob > threshold

    return prob