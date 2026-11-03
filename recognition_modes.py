"""Character-set constraints applied to CTC model probabilities before decoding.

Both modes share the pretrained weights. They have separate admissible symbol
sets, with unchanged original probabilities: restrictions cannot turn a weak
guess into a falsely confident result. No text substitution or label lookup.
"""
from __future__ import annotations

import string
import unicodedata
import numpy as np

MODE_LABELS = {"chinese": "汉字", "alphanumeric": "字母／数字"}
VALID_MODES = ("auto", *MODE_LABELS)
COMMON_PUNCTUATION = frozenset(string.punctuation + "，。！？；：“”‘’（）《》〈〉【】「」『』〔〕、…—·～")


def validate_mode(mode):
    if not isinstance(mode, str) or mode not in VALID_MODES:
        raise ValueError("请选择汉字或字母／数字识别模式。")
    return mode


def is_han(char):
    code = ord(char)
    return (0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
            or 0x20000 <= code <= 0x323AF or char == "〇")


def is_alphanumeric(char):
    normalized = unicodedata.normalize("NFKC", char)
    return bool(normalized) and all(c in string.ascii_letters + string.digits for c in normalized)


def allows_symbol(char, mode):
    if char.isspace() or char in COMMON_PUNCTUATION:
        return True
    return is_han(char) if mode == "chinese" else is_alphanumeric(char)


class ModeCTCDecoder:
    """Wrap RapidOCR's installed decoder without changing its dictionary indices.

    The blank token remains available and repeated tokens follow the upstream
    CTC rules. A restricted prediction keeps the original softmax probability;
    normal confidence filtering can still reject a weak alternative.
    """

    def __init__(self, base_decoder, mode):
        self.mode = validate_mode(mode)
        self.base_decoder = base_decoder
        ignored = set(base_decoder.get_ignored_tokens())
        self.allowed = np.array([
            mode == "auto" or index in ignored or bool(token) and all(allows_symbol(c, mode) for c in token)
            for index, token in enumerate(base_decoder.character)
        ], dtype=bool)
        self.saw_other_script = False

    def __getattr__(self, name):
        return getattr(self.base_decoder, name)

    def __call__(self, preds, return_word_box=False, **kwargs):
        if self.mode == "auto":
            return self.base_decoder(preds, return_word_box, **kwargs)
        if preds.ndim != 3 or preds.shape[-1] != len(self.allowed):
            raise ValueError("识别模型的字符表与输出维度不匹配。")
        # This cheap unrestricted decode only detects a possible mode mismatch;
        # it never supplies/replaces the returned mode-constrained prediction.
        original_lines, _ = self.base_decoder(preds, False, **kwargs)
        for text, _ in original_lines:
            if any(not allows_symbol(char, self.mode) for char in text):
                self.saw_other_script = True
        restricted = preds.copy()
        restricted[..., ~self.allowed] = -np.inf
        return self.base_decoder(restricted, return_word_box, **kwargs)
