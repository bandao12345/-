"""Geometry-only review candidates for separated handwritten Han components.

This module never maps recognized words to other words.  It preserves every
stroke, changes only horizontal aspect ratio and asks the same OCR model to
recognize the resulting image again.  Different geometry views can disagree
about whether two shapes are radicals or independent characters.  Therefore
even a multi-view candidate is presented for review, never silently applied.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median

import numpy as np
from PIL import Image

from recognition_modes import is_han
from whiteboard_layout import _ink_mask
from ocr_engine_v6 import _read_order


VIEW_SCALES = (0.6, 0.5, 0.4)
MAX_REVIEW_LINES = 3


def _line_geometry(mask, line):
    """Describe original pixels in a source-coordinate line rectangle."""
    points = np.asarray(line.get("box", ()), dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    height, width = mask.shape
    left = max(0, int(np.floor(points[:, 0].min())))
    top = max(0, int(np.floor(points[:, 1].min())))
    right = min(width, int(np.ceil(points[:, 0].max())))
    bottom = min(height, int(np.ceil(points[:, 1].max())))
    if right <= left or bottom <= top:
        return None
    ys, xs = np.nonzero(mask[top:bottom, left:right])
    if len(xs) < 16:
        return None
    ink_width = int(xs.max() - xs.min() + 1)
    ink_height = int(ys.max() - ys.min() + 1)
    han_count = sum(is_han(char) for char in str(line.get("text", "")))
    # Small printed/photo text and a lone predicted character do not need a
    # component-joining review.  Do not turn every Chinese line into an ensemble.
    if ink_height < 60 or han_count < 2:
        return None
    aspect = ink_width / max(1, ink_height)
    per_character = aspect / han_count
    original_score = float(line.get("confidence", 0))
    # A low-scoring wide glyph sequence or unusually narrow predicted glyphs
    # provides geometry/model evidence of a possible component split. Neither
    # condition contains character identities or a word vocabulary.
    reason = None
    if original_score < 0.88:
        reason = "original_recognition_uncertain"
    elif per_character < 0.82:
        reason = "predicted_glyphs_narrow_relative_to_ink_height"
    if reason is None:
        return None
    return {
        "region": (left, top, right, bottom),
        "ink_width": ink_width,
        "ink_height": ink_height,
        "ink_aspect_ratio": round(aspect, 4),
        "predicted_character_aspect": round(per_character, 4),
        "reason": reason,
    }


def propose_han_structure(engine, prepared: Image.Image, lines: list[dict]):
    """Return optional full-text review candidates without changing ``lines``.

    The caller must hold the recognizer lock and install its Chinese CTC
    decoder.  Only clean-board writing is considered.  A candidate needs two
    agreeing image views and a usable original model probability.  These are
    evidence for review, not a measurement of recognition accuracy.
    """
    ink = _ink_mask(prepared)
    if ink is None:
        return []
    mask, _ = ink
    ordered, raw_text = _read_order(lines)
    candidates = []
    reviewed = 0
    for line_index, line in enumerate(ordered):
        geometry = _line_geometry(mask, line)
        if geometry is None:
            continue
        if reviewed >= MAX_REVIEW_LINES:
            break
        reviewed += 1
        region = geometry.pop("region")
        crop = prepared.crop(region)
        groups = defaultdict(list)
        for scale in VIEW_SCALES:
            transformed = crop.resize(
                (max(8, round(crop.width * scale)), crop.height),
                Image.Resampling.LANCZOS,
            )
            output = engine(transformed, use_det=False, use_cls=False, use_rec=True)
            texts = getattr(output, "txts", None)
            scores = getattr(output, "scores", None)
            if texts is None or scores is None:
                continue
            for text, score in zip(texts, scores):
                text, score = str(text).strip(), float(score)
                if (not text or text == line["text"] or not np.isfinite(score)
                        or score < 0.8 or not any(is_han(char) for char in text)):
                    continue
                groups[text].append((scale, min(1.0, max(0.0, score))))
        for text, evidence in sorted(groups.items(), key=lambda item: -len(item[1])):
            # One unusual aspect ratio is not sufficient evidence for a review
            # suggestion.  Repeated views remain correlated and are not proof.
            if len(evidence) < 2:
                continue
            proposal_lines = [dict(item) for item in ordered]
            proposal_lines[line_index]["text"] = text
            _, complete_text = _read_order(proposal_lines)
            if complete_text == raw_text:
                continue
            candidates.append({
                "text": complete_text,
                "line_text": text,
                "original_text": raw_text,
                "original_line_text": line["text"],
                "line_index": line_index,
                "confidence": round(float(median(score for _, score in evidence)), 6),
                "support": len(evidence),
                "view_scales": [scale for scale, _ in evidence],
                "box": line["box"],
                "geometry": geometry,
                "reason": "保留笔画，仅调整左右部件的水平比例后，多个尺度得到相同候选；也可能原本就是独立汉字，请结合原笔迹确认。",
                "auto_applied": False,
            })
            break
    return candidates
