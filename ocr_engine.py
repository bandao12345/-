"""Production OCR: PP-OCRv6 medium, preserving original handwriting.

The previous PP-OCRv5 implementation remains in ocr_engine_v5.py and models/ocr.
No dictionary substitutions or user-example answers are used at inference time.
"""
from __future__ import annotations
import re
import time
import threading

from ocr_engine_v6 import (
    ENGINE_NAME, MODEL_FILES, UnifiedRecognizer as _V6Recognizer,
    _read_order, classify_script,
)
from whiteboard_layout import recognize_whiteboard
from recognition_modes import MODE_LABELS, ModeCTCDecoder, validate_mode
from han_layout import propose_han_structure


class UnifiedRecognizer(_V6Recognizer):
    """Use complete geometric rows on clean boards, detection on complex images."""

    def __init__(self, model_dir=None):
        super().__init__(model_dir)
        # Every decoder swap and inference belongs to one serial transaction.
        # A reentrant lock also protects the existing scene-detection fallback.
        self._lock = threading.RLock()

    @staticmethod
    def _enhance_thin_strokes(image):
        # Stroke dilation improved a few controls but destroyed detached F bars
        # in actual writing. Preserve original strokes in every production route.
        return image, False

    def status(self):
        result = super().status()
        result["pipeline_version"] = "2.3"
        result["layout"] = "automatic complete rows / scene detection"
        result["stroke_dilation"] = False
        result["supported_modes"] = [{"id": key, "label": label} for key, label in MODE_LABELS.items()]
        result["mode_constraint"] = "CTC character probabilities, before decoding"
        result["han_structure"] = "Chinese mode only: geometry review candidates; original text preserved"
        return result

    def predict(self, image, recognition_mode="auto"):
        started = time.perf_counter()
        mode = validate_mode(recognition_mode)
        if self._engine is None:
            raise RuntimeError(self._error or "文字识别引擎不可用")
        with self._lock:
            original = self._engine.text_rec.postprocess_op
            constrained = None if mode == "auto" else ModeCTCDecoder(original, mode)
            if constrained is not None:
                self._engine.text_rec.postprocess_op = constrained
            try:
                result = self._predict_image(image)
                # The primary OCR result remains unchanged. Chinese review
                # candidates are re-inferred from transformed pixels, never a
                # word/radical substitution table. Keep candidate-only decoder
                # warnings from being mistaken for an original mode mismatch.
                if mode == "chinese":
                    result["raw_text"] = result["text"]
                    original_mode_mismatch = constrained.saw_other_script
                    prepared, _ = self._prepare(image)
                    reliable_lines = [line for line in result["lines"] if line["confidence"] >= 0.35]
                    result["structure_candidates"] = propose_han_structure(
                        self._engine, prepared, reliable_lines)
                    constrained.saw_other_script = original_mode_mismatch
                    if result["structure_candidates"]:
                        result["warnings"].append(
                            "部分左右部件可能属于一个整字，已提供结构候选；原始识别保留，请结合笔迹人工确认。")
            finally:
                self._engine.text_rec.postprocess_op = original
            result["recognition_mode"] = mode
            if mode != "auto":
                # Full-row rec-only bypasses RapidOCR's scene score filter.
                # Apply the same reliability floor to constrained alternatives,
                # so a foreign glyph cannot become a confident forced guess.
                reliable = [line for line in result["lines"] if line["confidence"] >= 0.35]
                if len(reliable) < len(result["lines"]):
                    lines, text = _read_order(reliable)
                    count = sum(len(line["text"]) for line in lines)
                    result.update(lines=lines, text=text, script=classify_script(text),
                                  confidence=round(sum(line["confidence"] * len(line["text"]) for line in lines) / max(1, count), 6))
                    result["warnings"].append("部分区域的候选字符置信度过低，未作为可靠文字输出；请检查所选模式。")
            if constrained is not None and constrained.saw_other_script:
                result["warnings"].append(
                    f"当前为{MODE_LABELS[mode]}模式，部分笔迹可能属于另一类文字；请核对所选模式及识别结果。")
            if mode != "auto" and not result["text"]:
                result["warnings"].append(f"{MODE_LABELS[mode]}模式下没有可靠结果，请检查模式或书写更清晰的文字。")
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result

    def _predict_image(self, image):
        started = time.perf_counter()
        if self._engine is None:
            raise RuntimeError(self._error or "文字识别引擎不可用")
        prepared, _ = self._prepare(image)
        with self._lock:
            board = recognize_whiteboard(self._engine, prepared)
        if board is None or not board["lines"]:
            # Scene text still needs the detector. This fallback also retains
            # original strokes because _enhance_thin_strokes is overridden.
            result = super().predict(prepared)
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            result["recognition_route"] = "scene_detection_original_strokes"
            return result

        lines, text = _read_order(board["lines"])
        length = sum(len(line["text"]) for line in lines)
        confidence = sum(line["confidence"] * len(line["text"]) for line in lines) / max(length, 1)
        warnings = []
        if any(line["confidence"] < 0.8 for line in lines):
            warnings.append("部分文字置信度较低，请结合原笔迹核对。")
        if len(text.strip()) == 1 and text.strip() in "0Oo1Il|":
            warnings.append("该字符与 0/O、1/I/l 外形接近，请结合上下文核对。")
        if re.search(r"[A-Za-z]", text) and re.search(r"[0-9]", text):
            warnings.append("字母和数字混写时，O/0、I/1、b/6 等形近字符可能混淆，请核对。")
        return {
            "text": text,
            "confidence": round(confidence, 6),
            "script": classify_script(text),
            "lines": lines,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "warnings": warnings,
            "engine": ENGINE_NAME,
            "recognition_route": "clean_board_complete_rows_original_strokes",
        }

