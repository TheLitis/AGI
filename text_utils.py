"""
Minimal text utilities for instruction-conditioned experiments.

This project avoids heavy NLP dependencies. We use a stable hashing trick:
token -> id in [1, vocab_size-1], with 0 reserved for padding.
"""

from __future__ import annotations

import re
import zlib
from typing import Iterable, List

import numpy as np


_SPLIT_CHARS = "_-/\\|:.,;()[]{}<>\"'"


def tokenize_text(text: str) -> List[str]:
    """
    Lightweight tokenizer that works for Latin/Cyrillic without extra deps.
    """
    s = (text or "").lower()
    for ch in _SPLIT_CHARS:
        s = s.replace(ch, " ")
    tokens = re.findall(r"\w+", s, flags=re.UNICODE)
    return [t for t in tokens if t]


def hash_tokens_to_ids(tokens: Iterable[str], vocab_size: int) -> List[int]:
    """
    Map tokens to deterministic ids in [1, vocab_size-1] (0 is reserved for PAD).
    """
    vocab_size = int(vocab_size)
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2 (0 is PAD)")
    out: List[int] = []
    mod = vocab_size - 1
    for tok in tokens:
        h = zlib.crc32(tok.encode("utf-8")) & 0xFFFFFFFF
        out.append(int(h % mod) + 1)
    return out


def hash_text_to_ids(text: str, max_len: int, vocab_size: int) -> np.ndarray:
    """
    Tokenize + hash into a fixed-length int64 array.
    """
    max_len = int(max_len)
    if max_len <= 0:
        return np.zeros((0,), dtype=np.int64)
    ids = hash_tokens_to_ids(tokenize_text(text), vocab_size=vocab_size)[:max_len]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)

