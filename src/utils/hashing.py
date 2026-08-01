"""Shared deterministic hashing utilities for data splitting.

Used by both the live training loader (`AuroraLoader.split_train_val`) and the
offline QC/dedup pipeline (`make_splits.py`'s hash-split fallback) so that the
same document is assigned to the same split by either code path.
"""

import hashlib


def deterministic_split_value(text: str, num_chars: int = 500) -> float:
    """Map a document's leading text to a deterministic float in [0, 1).

    MD5 is fine here (not security-critical, just needs a uniform distribution
    over documents). Uses the full 128-bit digest for split-value resolution
    rather than a truncated prefix, so callers get consistent results without
    each re-implementing the same hash-to-float conversion slightly differently.

    Args:
        text: Document text (only the first `num_chars` characters are hashed).
        num_chars: Number of leading characters to hash.

    Returns:
        A float in [0, 1), deterministic for a given (text, num_chars) pair.
    """
    digest = hashlib.md5(text[:num_chars].encode("utf-8")).hexdigest()
    return int(digest, 16) / 16**32
