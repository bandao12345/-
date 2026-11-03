"""MNIST CPU inference and bounded image decoding.

Net and its trained weights follow the Huawei ModelArts MNIST example:
https://support.huaweicloud.com/qs-modelarts/modelarts_06_0035.html
The web application owns a DigitRecognizer instance; models are not shared
between application instances.
"""
from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
import threading
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from model import Net

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist.pt"
MEAN = 0.1307
STD = 0.3081
INPUT_SIZE = 28
INNER_SIZE = 20
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_DIMENSION = 4096
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
BLANK_THRESHOLD = 8
LOW_CONFIDENCE_THRESHOLD = 0.80

torch.set_num_threads(2)


class ImageValidationError(ValueError):
    """An invalid input that the API can report as a client error."""


def _check_dimensions(image):
    width, height = image.size
    if width < 1 or height < 1:
        raise ImageValidationError("图像尺寸无效")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageValidationError("图像宽高不能超过 4096 像素")


def validate_image_bytes(data):
    """Read actual PNG/JPEG/WEBP, checking limits before pixel decoding."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ImageValidationError("图像文件必须是字节数据")
    if not data:
        raise ImageValidationError("图像数据为空")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageValidationError("图像文件不能超过 2MB")
    data = bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as header:
            if header.format not in ALLOWED_FORMATS:
                raise ImageValidationError("仅支持真实 PNG、JPEG 或 WEBP 图像")
            _check_dimensions(header)
            header.verify()
        with Image.open(io.BytesIO(data)) as decoded:
            _check_dimensions(decoded)
            decoded.load()
            return decoded.copy()
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError("图像文件无效或已损坏") from error


def decode_base64_image(value):
    """Strictly decode a data URL or canonical raw Base64 string."""
    if not isinstance(value, str):
        raise ImageValidationError("图像数据必须是 Base64 字符串")
    encoded = value.strip()
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise ImageValidationError("图像 data URL 格式无效")
        header, encoded = encoded.split(",", 1)
        if header.lower() not in {
            "data:image/png;base64", "data:image/jpeg;base64",
            "data:image/jpg;base64", "data:image/webp;base64",
        }:
            raise ImageValidationError("仅支持 PNG、JPEG 或 WEBP 的 Base64 图像")
    if not encoded:
        raise ImageValidationError("图像数据为空")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageValidationError("图像文件不能超过 2MB")
    try:
        decoded = base64.b64decode(encoded, validate=True)
        if base64.b64encode(decoded).decode("ascii") != encoded:
            raise ValueError("noncanonical Base64")
    except (ValueError, binascii.Error) as error:
        raise ImageValidationError("Base64 图像编码无效") from error
    return validate_image_bytes(decoded)


def allowed_filename(filename):
    return isinstance(filename, str) and Path(filename).suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS


def _flatten_alpha(image):
    """Composite transparent ink against a contrasting background.

    Hidden RGB pixels do not determine the background: white ink on transparent
    pixels needs a black background, while black ink needs a white background.
    Converting LA/P through RGBA also avoids unsupported LA alpha composition.
    """
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return image.convert("L")
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32)
    if float(alpha.max()) < BLANK_THRESHOLD:
        raise ImageValidationError("图像为空白，请先写一个数字")
    if np.any(alpha < 255):
        gray = np.asarray(rgba.convert("L"), dtype=np.float32)
        visible = alpha >= max(BLANK_THRESHOLD, float(alpha.max()) * 0.25)
        ink_luminance = float(np.average(gray[visible], weights=alpha[visible]))
        background = 0 if ink_luminance > 127 else 255
        base = Image.new("RGBA", rgba.size, (background, background, background, 255))
        return Image.alpha_composite(base, rgba).convert("L")
    return rgba.convert("L")


def _center_mass(canvas):
    """Shift the intensity centroid without wrapping pixels or clipping ink."""
    total = float(canvas.sum())
    if total <= 0:
        return canvas
    rows, cols = np.indices(canvas.shape)
    center_y = float((rows * canvas).sum() / total)
    center_x = float((cols * canvas).sum() / total)
    ink_rows, ink_cols = np.nonzero(canvas)
    dx = int(round((INPUT_SIZE - 1) / 2 - center_x))
    dy = int(round((INPUT_SIZE - 1) / 2 - center_y))
    dx = int(np.clip(dx, -int(ink_cols.min()), INPUT_SIZE - 1 - int(ink_cols.max())))
    dy = int(np.clip(dy, -int(ink_rows.min()), INPUT_SIZE - 1 - int(ink_rows.max())))
    shifted = np.zeros_like(canvas)
    src_y0, src_y1 = max(0, -dy), min(INPUT_SIZE, INPUT_SIZE - dy)
    src_x0, src_x1 = max(0, -dx), min(INPUT_SIZE, INPUT_SIZE - dx)
    shifted[src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = canvas[src_y0:src_y1, src_x0:src_x1]
    return shifted


def preprocess(image):
    """Return the normalized tensor and the exact 28×28 pixels fed to Net.

    Existing 28×28 MNIST samples keep their geometry. Other inputs are cropped,
    resized with aspect ratio preserved inside 20×20, and centered in 28×28.
    """
    if not isinstance(image, Image.Image):
        raise ImageValidationError("输入必须是有效图像")
    _check_dimensions(image)
    try:
        oriented = ImageOps.exif_transpose(image)
        gray = _flatten_alpha(oriented)
        raw = np.asarray(gray, dtype=np.float32)
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError("无法读取图像内容") from error
    if float(raw.max() - raw.min()) < BLANK_THRESHOLD:
        raise ImageValidationError("图像为空白或纯色，请先写一个数字")

    border = np.concatenate((raw[0, :], raw[-1, :], raw[:, 0], raw[:, -1]))
    background = float(np.median(border))
    ink = np.clip(background - raw if background > 127 else raw - background, 0, 255)
    peak = float(ink.max())
    if peak < BLANK_THRESHOLD:
        raise ImageValidationError("未检测到清晰笔画，请重新书写")

    if raw.shape == (INPUT_SIZE, INPUT_SIZE):
        preview = ink.astype(np.uint8)
    else:
        mask = ink > max(BLANK_THRESHOLD, peak * 0.12)
        positions = np.argwhere(mask)
        if not len(positions):
            raise ImageValidationError("未检测到数字笔画")
        y0, x0 = positions.min(axis=0)
        y1, x1 = positions.max(axis=0) + 1
        cropped = ink[y0:y1, x0:x1]
        height, width = cropped.shape
        scale = INNER_SIZE / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = np.asarray(Image.fromarray(cropped.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        top = (INPUT_SIZE - size[1]) // 2
        left = (INPUT_SIZE - size[0]) // 2
        canvas[top:top + size[1], left:left + size[0]] = resized
        preview = _center_mass(canvas).astype(np.uint8)
    normalized = (preview.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0), preview


def _array_to_png_data_uri(array):
    output = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def make_thumbnail(image):
    return _array_to_png_data_uri(preprocess(image)[1])


class DigitRecognizer:
    """One independently loaded, CPU-only model for one application instance."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self._model = None
        self._error = None
        self._lock = threading.RLock()

    def _load_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                self._error = "模型文件不存在，请先运行训练脚本"
                raise FileNotFoundError(self._error)
            try:
                state = torch.load(self.model_path, map_location="cpu", weights_only=True)
                model = Net()
                model.load_state_dict(state)
                model.eval()
            except Exception as error:
                self._error = "模型文件无法加载，请检查权重或重新训练"
                raise RuntimeError(self._error) from error
            self._model = model
            self._error = None
            return model

    def status(self):
        exists = self.model_path.is_file()
        try:
            self._load_model()
            ready = True
        except (FileNotFoundError, RuntimeError):
            ready = False
        result = {"ready": ready, "model_exists": exists,
                  "model_loaded": self._model is not None, "device": "cpu"}
        if self._error:
            result["error"] = self._error
        return result

    def predict(self, image):
        started = time.perf_counter()
        tensor, preview = preprocess(image)
        model = self._load_model()
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor)[0], dim=0).tolist()
        ranked = sorted(range(10), key=probabilities.__getitem__, reverse=True)
        digit = ranked[0]
        confidence = float(probabilities[digit])
        warnings = []
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append("识别置信度较低，请把数字写大、写清楚后重试")
        return {
            "digit": digit,
            "confidence": confidence,
            "probabilities": [float(p) for p in probabilities],
            "top3": [{"digit": i, "probability": float(probabilities[i])} for i in ranked[:3]],
            "preview": _array_to_png_data_uri(preview),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "warnings": warnings,
        }
