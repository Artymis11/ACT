"""Shared local RGB preprocessing for ACT datasets and observation uploads.

The geometry follows OpenPI's client resize_with_pad convention.
Reference: https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/image_tools.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
from PIL import Image


CONTRACT_FILENAME = "image_preprocessing.json"
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
PILLOW_VERSION = "12.3.0"


def preprocessing_contract():
    return {
        "schema_version": 1,
        "height": TARGET_HEIGHT,
        "width": TARGET_WIDTH,
        "channels": 3,
        "color_order": "RGB",
        "dtype": "uint8",
        "geometry": "aspect_preserving_resize_center_zero_pad",
        "interpolation": "PIL.Image.Resampling.BILINEAR",
        "resized_dimension_rounding": "floor",
        "padding_offset_rounding": "floor",
        "padding_rgb": [0, 0, 0],
        "pillow_version": PILLOW_VERSION,
        "transport": "lerobot_pickle_numpy_uint8_hwc",
        "lossy_transport_encoding": False,
    }


def preprocess_rgb(image):
    """Resize one HWC RGB uint8 frame before local storage or network upload."""
    if PIL.__version__ != PILLOW_VERSION:
        raise ValueError(f"Image preprocessing requires Pillow {PILLOW_VERSION}, found {PIL.__version__}.")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected a numpy uint8 RGB frame.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError(f"Expected a nonempty HWC RGB frame, received {image.shape}.")
    if image.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3):
        return np.ascontiguousarray(image)
    height, width = image.shape[:2]
    ratio = max(width / TARGET_WIDTH, height / TARGET_HEIGHT)
    resized_width = max(1, int(width / ratio))
    resized_height = max(1, int(height / ratio))
    resized = Image.fromarray(np.ascontiguousarray(image)).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR,
    )
    output = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    left = (TARGET_WIDTH - resized_width) // 2
    top = (TARGET_HEIGHT - resized_height) // 2
    output[top:top + resized_height, left:left + resized_width] = np.asarray(resized)
    return output


def contract_digest(contract=None):
    value = preprocessing_contract() if contract is None else contract
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_contract(directory):
    path = Path(directory) / CONTRACT_FILENAME
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract != preprocessing_contract():
        raise ValueError(f"Image preprocessing contract differs from this implementation: {path}")
    if PIL.__version__ != PILLOW_VERSION:
        raise ValueError(f"Image preprocessing requires Pillow {PILLOW_VERSION}.")
    return contract


def validate_model_contract(directory):
    directory = Path(directory)
    contract = load_contract(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    cameras = {key: feature for key, feature in config["input_features"].items()
               if feature["type"] == "VISUAL"}
    if config.get("type") != "act" or not cameras:
        raise ValueError("Expected an ACT checkpoint with visual features.")
    for key, feature in cameras.items():
        if list(feature["shape"]) != [3, TARGET_HEIGHT, TARGET_WIDTH]:
            raise ValueError(f"Model image shape is inconsistent with its preprocessing contract: {key}")
    return contract
