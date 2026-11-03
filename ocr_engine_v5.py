"""Offline multilingual OCR for INKLAB.

PP-OCRv5 weights are upstream pretrained models, not the project's MNIST model.
Inference is local CPU only. See models/ocr/manifest.json for sources and hashes.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ENGINE_NAME = "PP-OCRv5 · RapidOCR / ONNX Runtime"
MODEL_FILES = {
    "ch_PP-OCRv5_det_mobile.onnx": "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    "ch_PP-OCRv5_rec_server.onnx": "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
}


def classify_script(text: str) -> str:
    """Describe the actual result; never use this label to select a model."""
    scripts = set()
    for char in text:
        if char.isdigit():
            scripts.add("digits")
        elif "\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U0003134f":
            scripts.add("han")
        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            scripts.add("latin")
    return next(iter(scripts)) if len(scripts) == 1 else "mixed"


def _read_order(lines: list[dict]) -> tuple[list[dict], str]:
    """Group overlapping horizontal text lines, then sort left to right."""
    if not lines:
        return [], ""
    rows = []
    for line in sorted(lines, key=lambda item: min(p[1] for p in item["box"])):
        ys = [p[1] for p in line["box"]]
        top, bottom = min(ys), max(ys)
        center = (top + bottom) / 2
        height = max(1, bottom - top)
        match = None
        for row in rows:
            overlap = min(bottom, row["bottom"]) - max(top, row["top"])
            if overlap >= 0.45 * min(height, row["height"]):
                match = row
                break
        if match is None:
            rows.append({"top": top, "bottom": bottom, "height": height,
                         "center": center, "items": [line]})
        else:
            match["items"].append(line)
            match["top"] = min(match["top"], top)
            match["bottom"] = max(match["bottom"], bottom)
    rows.sort(key=lambda row: row["top"])
    ordered, texts = [], []
    for row in rows:
        row["items"].sort(key=lambda item: min(p[0] for p in item["box"]))
        ordered.extend(row["items"])
        pieces = [row["items"][0]["text"]]
        for previous, item in zip(row["items"], row["items"][1:]):
            # Preserve separate Latin words. Chinese/digit fragments concatenate.
            boundary_latin = re.search(r"[A-Za-z]$", previous["text"]) and re.match(r"[A-Za-z]", item["text"])
            pieces.append((" " if boundary_latin else "") + item["text"])
        texts.append("".join(pieces))
    return ordered, "\n".join(texts)


class UnifiedRecognizer:
    """One model route handles digits, Latin letters and Chinese together.

    A lock is required because RapidOCR updates its call options on each request.
    Model downloads never occur at inference time: every model path is explicit.
    """
    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir or Path(__file__).parent / "models" / "ocr")
        self._engine = None
        self._error = ""
        self._lock = threading.Lock()
        missing = [name for name in MODEL_FILES if not (self.model_dir / name).is_file()]
        if missing:
            self._error = "本地 OCR 模型文件缺失，请按 README 准备完整模型。"
            return
        try:
            for filename, expected in MODEL_FILES.items():
                digest = hashlib.sha256()
                with (self.model_dir / filename).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    self._error = "本地 OCR 模型校验失败，请重新下载官方模型。"
                    return
            import onnxruntime as ort
            from omegaconf import flag_override
            from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.log_severity_level = 4
            # Flush denormals to avoid data-dependent 20–50 s CPU slow paths.
            # This changes numerical handling, not model weights. Regression
            # fixtures verify actual decoded text after enabling this option.
            options.add_session_config_entry("session.set_denormal_as_zero", "1")
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            rec_session = ort.InferenceSession(
                str(self.model_dir / "ch_PP-OCRv5_rec_server.onnx"),
                sess_options=options, providers=["CPUExecutionProvider"])

            class LocalRapidOCR(RapidOCR):
                def _load_config(self, config_path, params):
                    cfg = super()._load_config(config_path, params)
                    # RapidOCR explicitly supports a supplied ORT session.
                    with flag_override(cfg, "allow_objects", True):
                        cfg.Rec.session = rec_session
                    return cfg

            self._engine = LocalRapidOCR(params={
                "Global.log_level": "error",
                "Global.text_score": 0.35,
                "Global.use_cls": False,
                "Global.max_side_len": 1600,
                "Global.use_vertical_padding": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.onnxruntime.use_cuda": False,
                "EngineConfig.onnxruntime.use_dml": False,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_path": str(self.model_dir / "ch_PP-OCRv5_det_mobile.onnx"),
                "Det.limit_type": "max",
                "Det.limit_side_len": 960,
                # PP-OCRv5 detector training normalization, from PaddleOCR config.
                "Det.mean": [0.485, 0.456, 0.406],
                "Det.std": [0.229, 0.224, 0.225],
                "Cls.model_path": str(self.model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CH,
                "Rec.model_type": ModelType.SERVER,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_path": str(self.model_dir / "ch_PP-OCRv5_rec_server.onnx"),
                "Rec.rec_batch_num": 1,
            })
        except ImportError:
            self._error = "OCR 依赖未安装，请使用升级版的独立 Python 环境。"
        except Exception:
            self._error = "OCR 引擎初始化失败，请检查模型、运行库和可用内存。"

    def status(self) -> dict:
        return {
            "ready": self._engine is not None,
            "engine": ENGINE_NAME,
            "device": "cpu",
            "threads": 2,
            "model_version": "PP-OCRv5",
            "source_url": "https://github.com/RapidAI/RapidOCR",
            "license": "Apache-2.0",
            **({"error": self._error} if self._error else {}),
        }

    @staticmethod
    def _prepare(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
        if not isinstance(image, Image.Image):
            raise ValueError("请提供有效图片。")
        if not image.width or not image.height or max(image.size) > 4096:
            raise ValueError("图片宽、高必须在 1–4096 像素之间。")
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            visible = np.asarray(rgba)
            alpha = visible[:, :, 3] > 0
            if not np.any(alpha):
                raise ValueError("图片为空白，请先书写或上传有文字的图片。")
            brightness = float(np.mean(visible[:, :, :3][alpha]))
            background = (255, 255, 255, 255) if brightness < 128 else (0, 0, 0, 255)
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, background), rgba)
        image = image.convert("RGB")
        gray = np.asarray(image.convert("L"))
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        if float(np.median(border)) < 100:
            image = ImageOps.invert(image)
            gray = 255 - gray
            border = 255 - border
        contrast = int(gray.max()) - int(gray.min())
        background = float(np.median(border))
        foreground = np.abs(gray.astype(np.float32) - background) > 25
        if contrast < 20 or int(foreground.sum()) < 8:
            raise ValueError("图片为空白或笔迹太浅，请先书写清晰文字。")
        ys, xs = np.nonzero(foreground)
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        padding = max(8, min(32, round((bottom - top) * 0.12)))
        crop_box = (max(0, left - padding), max(0, top - padding),
                    min(image.width, right + padding), min(image.height, bottom + padding))
        return image, crop_box

    @staticmethod
    def _enhance_thin_strokes(image: Image.Image) -> tuple[Image.Image, bool]:
        """Improve thin writing on clean paper without thickening small text.

        Decisions use image geometry only. The distance-transform estimate of
        stroke width is compared with connected-component height; neither
        recognized text nor language controls enhancement.
        """
        import cv2
        gray = np.asarray(image.convert("L"))
        if float(np.mean(gray > 245)) < 0.7:
            return image, False
        mask = (gray < 180).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        components = stats[1:]
        components = components[components[:, cv2.CC_STAT_AREA] >= 16]
        if len(components) == 0:
            return image, False
        glyph_height = float(np.percentile(components[:, cv2.CC_STAT_HEIGHT], 75))
        if glyph_height < 60:
            return image, False
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        stroke_width = 2 * float(np.median(distance[mask > 0]))
        if stroke_width / glyph_height >= 0.035:
            return image, False
        kernel_size = 3 if glyph_height < 220 else 5
        enhanced = ImageOps.autocontrast(image.convert("L"))
        return enhanced.filter(ImageFilter.MinFilter(kernel_size)), True

    @staticmethod
    def _lines(output, offset: tuple[int, int], image_size: tuple[int, int],
               fallback_box: tuple[int, int, int, int]) -> list[dict]:
        txts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if txts is None or scores is None:
            return []
        boxes = getattr(output, "boxes", None)
        lines = []
        width, height = image_size
        for index, (text, score) in enumerate(zip(txts, scores)):
            text = str(text).strip()
            score = float(score)
            if not text or not np.isfinite(score) or score < 0.25:
                continue
            if boxes is not None and index < len(boxes):
                box = [[round(min(width, max(0.0, float(x) + offset[0])), 1),
                        round(min(height, max(0.0, float(y) + offset[1])), 1)]
                       for x, y in boxes[index]]
            else:
                left, top, right, bottom = fallback_box
                box = [[left, top], [right, top], [right, bottom], [left, bottom]]
            lines.append({"text": text, "confidence": round(min(1, max(0, score)), 6), "box": box})
        return lines

    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        if self._engine is None:
            raise RuntimeError(self._error or "OCR 引擎不可用。")
        image, crop_box = self._prepare(image)
        image, enhanced = self._enhance_thin_strokes(image)
        if enhanced:
            # Recompute the foreground bounds after geometric stroke expansion.
            image, crop_box = self._prepare(image)
        left, top, right, bottom = crop_box
        cropped = image.crop(crop_box)
        warnings = []
        if enhanced:
            warnings.append("已自动增强较细笔迹，请核对易混淆字符。")
        with self._lock:
            detected = self._engine(cropped, use_det=True, use_cls=False, use_rec=True)
            lines = self._lines(detected, (left, top), image.size, crop_box)
            # A detector can omit isolated or very thin glyphs on a large board.
            # Retry the foreground crop directly, never switching language/model.
            ratio = cropped.width / cropped.height
            if not lines:
                fallback = self._engine(cropped, use_det=False, use_cls=False, use_rec=True)
                lines = self._lines(fallback, (left, top), image.size, crop_box)
                if lines:
                    warnings.append("已自动使用笔迹区域补充识别，请核对结果。")
            elif ratio <= 1.45 and max(line["confidence"] for line in lines) < 0.85 and sum(len(line["text"]) for line in lines) <= 2:
                fallback = self._engine(cropped, use_det=False, use_cls=False, use_rec=True)
                alternatives = self._lines(fallback, (left, top), image.size, crop_box)
                if alternatives and max(line["confidence"] for line in alternatives) > max(line["confidence"] for line in lines) + 0.025:
                    lines = alternatives
        lines, text = _read_order(lines)
        if not lines:
            warnings.append("未识别到可靠文字，请把文字写大、拉开间距或上传更清晰的图片。")
            confidence = 0.0
        else:
            length = sum(len(line["text"]) for line in lines)
            confidence = sum(line["confidence"] * len(line["text"]) for line in lines) / max(1, length)
            if any(line["confidence"] < 0.8 for line in lines):
                warnings.append("部分文字置信度较低，请人工核对并在记录中纠错。")
            if len(text.strip()) == 1 and text.strip() in "0Oo1Il|":
                warnings.append("该单字符与 0/O 或 1/I/l 外形接近，请结合上下文核对。")
        return {
            "text": text,
            "confidence": round(confidence, 6),
            "script": classify_script(text),
            "lines": lines,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "warnings": warnings,
            "engine": ENGINE_NAME,
        }

