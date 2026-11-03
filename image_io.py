"""Bounded image input for the unified OCR application."""
from __future__ import annotations
import base64
import binascii
import io
import numpy as np
from PIL import Image, ImageOps

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_DIMENSION = 4096
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


class ImageValidationError(ValueError):
    pass


def _dimensions(image):
    if not 1 <= image.width <= MAX_DIMENSION or not 1 <= image.height <= MAX_DIMENSION:
        raise ImageValidationError("图片宽高必须在 1 至 4096 像素之间")


def validate_image_bytes(data):
    if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
        raise ImageValidationError("图片内容为空或格式无效")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageValidationError("图片文件不能超过 2 MB")
    try:
        with Image.open(io.BytesIO(data)) as header:
            if header.format not in {"PNG", "JPEG", "WEBP"}:
                raise ImageValidationError("仅支持真实 PNG、JPEG 或 WEBP 图片")
            _dimensions(header)
            header.verify()
        with Image.open(io.BytesIO(data)) as decoded:
            _dimensions(decoded)
            decoded.load()
            return decoded.copy()
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError("图片损坏或无法解析，请重新选择") from error


def decode_base64_image(value):
    if not isinstance(value, str):
        raise ImageValidationError("图片必须是有效 Base64 字符串")
    value = value.strip()
    if value.startswith("data:"):
        if "," not in value:
            raise ImageValidationError("图片 data URL 格式无效")
        header, value = value.split(",", 1)
        if header.lower() not in {
            "data:image/png;base64", "data:image/jpeg;base64",
            "data:image/jpg;base64", "data:image/webp;base64",
        }:
            raise ImageValidationError("请选择 PNG、JPEG 或 WEBP 图片")
    if not value:
        raise ImageValidationError("图片内容为空")
    if len(value) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageValidationError("图片文件不能超过 2 MB")
    try:
        raw = base64.b64decode(value, validate=True)
        if base64.b64encode(raw).decode("ascii") != value:
            raise ValueError()
    except (ValueError, binascii.Error) as error:
        raise ImageValidationError("图片 Base64 编码无效") from error
    return validate_image_bytes(raw)


def prepare_image(image):
    """Use an upright, opaque image with dark ink on a light background."""
    _dimensions(image)
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        small = rgba.copy()
        small.thumbnail((512, 512))
        alpha = np.asarray(small.getchannel("A"), dtype=np.float32)
        if alpha.max() < 8:
            raise ImageValidationError("画面为空，请先书写或选择图片")
        visible = alpha >= max(8, float(alpha.max()) * 0.25)
        luminance = np.asarray(small.convert("L"), dtype=np.float32)
        ink_light = float(np.average(luminance[visible], weights=alpha[visible]))
        background = 0 if ink_light > 127 else 255
        image = Image.alpha_composite(
            Image.new("RGBA", rgba.size, (background,) * 3 + (255,)), rgba,
        ).convert("RGB")
    else:
        image = image.convert("RGB")
    small = image.convert("L")
    low, high = small.getextrema()
    if high - low < 8:
        raise ImageValidationError("画面为空白或纯色，请先书写文字")
    small.thumbnail((256, 256))
    gray = np.asarray(small)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) < 127:
        image = ImageOps.invert(image)
    return image


def make_preview(image):
    preview = image.copy()
    preview.thumbnail((1000, 700))
    png, jpeg = io.BytesIO(), io.BytesIO()
    preview.save(png, format="PNG", optimize=True)
    preview.save(jpeg, format="JPEG", quality=85)
    raw, mime = (png.getvalue(), "png") if png.tell() <= jpeg.tell() else (jpeg.getvalue(), "jpeg")
    return f"data:image/{mime};base64," + base64.b64encode(raw).decode("ascii")

