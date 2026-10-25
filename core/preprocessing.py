from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
import torch

TARGET_SIZE = (256, 256)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_image_as_rgb(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image, np.ndarray]
) -> np.ndarray:
    # Load input into numpy array
    if isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, Image.Image):
        arr = np.array(source)
    elif isinstance(source, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(source))
        arr = np.array(pil_img)
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        pil_img = Image.open(source)
        arr = np.array(pil_img)
    else:
        pil_img = Image.open(str(source))
        arr = np.array(pil_img)

    # Scale 16-bit or float arrays to uint8
    if arr.dtype in (np.uint16, np.int32, np.int16, np.float32, np.float64):
        min_val = float(arr.min())
        max_val = float(arr.max())
        val_range = max_val - min_val
        if val_range > 0:
            arr_u8 = np.clip(((arr.astype(np.float32) - min_val) / val_range) * 255.0, 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_u8 = arr.astype(np.uint8)

    # Convert to standard RGB
    if arr_u8.ndim == 2:
        return cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)
    elif arr_u8.ndim == 3:
        if arr_u8.shape[2] == 1:
            return cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)
        elif arr_u8.shape[2] == 4:
            return cv2.cvtColor(arr_u8, cv2.COLOR_RGBA2RGB)
        elif arr_u8.shape[2] == 3:
            return arr_u8
        return arr_u8[:, :, :3]
    raise ValueError(f"Unsupported array shape: {arr_u8.shape}")


def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    img_float = image.astype(np.float32) / 255.0
    return (img_float - IMAGENET_MEAN) / IMAGENET_STD


def hwc_to_tensor(image: np.ndarray) -> torch.Tensor:
    chw = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def preprocess_image(
    image: Union[np.ndarray, Image.Image, str, Path, bytes, io.BytesIO],
    size: tuple[int, int] = TARGET_SIZE,
) -> torch.Tensor:
    rgb_img = decode_image_as_rgb(image)
    resized = resize_image(rgb_img, size=size)
    normalized = normalize_image(resized)
    return hwc_to_tensor(normalized)
